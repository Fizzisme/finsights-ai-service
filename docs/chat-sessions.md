# Chat sessions và lịch sử hội thoại

Trạng thái: **đã triển khai**: nhiều lượt, lưu lịch sử, cửa sổ trượt và compaction
(rolling summary) theo cơ chế của BOA.

## Vấn đề

`POST /runs` xử lý một câu hỏi rồi quên. Người dùng không hỏi tiếp được
("bài thứ 2 nói gì?", "so với con số lúc nãy?"), và không có dữ liệu thật để xem
agent đã gọi tool gì cho câu hỏi nào.

Điểm khó riêng của Finsights: verifier chặn mọi con số/ngày không có trong tool
result **của lượt hiện tại**. Nếu lượt sau dùng lại con số đã kiểm chứng ở lượt
trước, verifier phải chấp nhận nó, nếu không agent sẽ bị chặn oan.

## Nguyên tắc (tham khảo BOA, rút gọn)

1. **Digest do code tạo, không dùng LLM.** Mỗi lượt để lại một digest ngắn: câu
   hỏi, câu trả lời (cắt ở 400 và 600 ký tự), các *fact đã kiểm chứng* (số, ngày,
   link). Không mang trace tool hay toàn văn tool result sang lượt sau.
2. **Ngân sách cố định.** Lịch sử đưa vào model tối đa `HISTORY_TOKENS = 2000`
   (ước lượng 3 ký tự/token vì tiếng Việt tốn token): bản tóm tắt các lượt cũ
   (`earlier`) cộng digest nguyên văn của các lượt chưa gấp (`turns`).
3. **Gấp khi vượt ngân sách (compaction).** Các lượt trước `KEEP_RECENT_TURNS = 3`
   lượt gần nhất được gấp vào rolling summary (xem mục Compaction). Gấp lỗi thì
   quay về cắt bớt: bỏ digest cũ nhất, luôn giữ lượt mới nhất. Lượt mới không bao
   giờ fail vì lịch sử dài.
4. **Fact cũ là nguồn hợp lệ.** Câu hỏi và fact của các lượt trong cửa sổ được
   verifier chấp nhận như tool result (kể cả làm input cho `calculate`). Chỉ lượt
   có `verified = true` mới để lại fact, nên con số chưa kiểm chứng không lan sang
   lượt sau.
5. **Lịch sử là dữ liệu, không phải lệnh.** Nó chứa chữ người dùng gõ và nội dung
   lấy từ bài báo, nên được bọc trong `<conversation_history>` (không thể tự đóng
   thẻ) và prompt coi nó như `<untrusted_input>`.

Code: `app/graph/history.py` (digest, cửa sổ, block, nguồn cho verifier),
`app/graph/compaction.py` + `app/prompts/compactor.md` (gấp).

## Mô hình dữ liệu (MySQL `finsights_ai`, Node 1)

Schema do [finsights-coredb](https://github.com/minhphu05/finsights-coredb)
quản lý (`mysql-node-1/schemas/ai.json`); service này không chạy migration. Giờ
lưu UTC (server `time_zone = UTC`).

`chat_sessions`: `id`, `session_uid` (id công khai), `user_uid`, `title` (câu hỏi
đầu, cắt 240 ký tự), `model_provider`, `session_status` (`active` | `deleted`),
`created_at`, `updated_at`, `closed_at`.

`chat_messages`: mỗi lượt là **hai dòng** cùng `turn`, `speaker` = `user` và
`assistant`.

| Cột | Dùng cho |
| --- | --- |
| `message_uid`, `session_id`, `speaker`, `message_text` | Nội dung |
| `message_status` | `complete` \| `failed` |
| `model_name`, `error_code` | Dòng assistant; `error_code` = `timeout` \| `no_answer` \| `provider_error` |
| `turn` | Số lượt trong phiên; unique `(session_id, turn, speaker)` chặn hai request ghi cùng một lượt |
| `ai_trace` (JSON) | Dòng assistant: `digest`, `tool_calls`, `verified`, `unsupported`, `latency_ms`, `cache_read_tokens`; khi lượt có gấp: `compaction` (kết quả) và `context` (checkpoint mới) |
| `prompt_tokens`, `completion_tokens` | Dòng assistant: tổng token mọi lần gọi model trong lượt |

`ai_trace.digest` ví dụ:

```json
{
  "request": "Vậy TCB nhiều hơn HPG bao nhiêu phần trăm?",
  "summary": "Tháng 10/2026, TCB có 296 bài, nhiều hơn 229 bài của HPG là 29,26%.",
  "facts": {"numbers": ["296", "229", "29,26"], "dates": [], "day_months": [], "links": []}
}
```

Lượt lỗi vẫn được lưu (`failed`) với digest không có fact, để người dùng thấy và
lượt sau biết lượt đó đã lỗi. Xoá phiên là xoá mềm.

## API

Mọi endpoint cần header `X-User-Id` (UUID) do gateway/backend đặt sau khi xác
thực. Service AI không tự xác thực, nên **không được mở thẳng ra internet**. Phiên
của người khác, `session_uid` sai định dạng, phiên đã xoá đều trả `404`.

| Method | Path | Việc |
| --- | --- | --- |
| `POST` | `/sessions` | Tạo phiên → `201` + `session_uid` |
| `GET` | `/sessions?limit=50` | Danh sách phiên, mới cập nhật trước |
| `GET` | `/sessions/{session_uid}` | Phiên + các lượt (`turn`, `speaker`, `text`, `status`, `created_at`) |
| `DELETE` | `/sessions/{session_uid}` | Xoá mềm → `204` |
| `POST` | `/sessions/{session_uid}/messages?profile=&tier=` | Hỏi một lượt; body `{question}`; trả như `/runs` + `session_uid`, `turn` |
| `POST` | `/sessions/{session_uid}/messages/stream?profile=&tier=` | Cùng lượt đó, trả tiến trình bằng Server-Sent Events (xem dưới) |
| `GET` | `/usage` | Quota token của user: đã dùng, hạn mức, lúc reset (xem [Quota](#quota-token)) |

Mã lỗi của `POST .../messages`: `401` thiếu/sai user, `404` không thấy phiên,
`429` hết quota token, `409` lượt bị ghi trùng, `502`/`504` agent lỗi (giống `/runs`), `503` DB lịch sử
không đọc được (agent không chạy). Nếu chỉ bước **lưu** lỗi, câu trả lời vẫn được
trả với `turn = null`.

`POST /runs` giữ nguyên (không lưu gì) cho eval và demo.

### Streaming (SSE)

Một lượt mất khoảng 10 giây. `.../messages/stream` chạy đúng lượt đó (cùng compaction,
lưu, mã lỗi) nhưng báo tiến trình trong lúc chờ. Câu trả lời **không** được stream từng
token: verifier chỉ kiểm được khi đã có cả câu, và một câu bị từ chối không được lọt tới
người dùng, kể cả trong chốc lát.

| Event | `data` | Khi nào |
| --- | --- | --- |
| `compacting` | `{}` | Lượt này gấp lịch sử cũ |
| `tool_start` | `{name, args}` | Model gọi một tool |
| `tool_end` | `{name, ok}` | Tool trả kết quả |
| `verifying` | `{}` | Model viết xong, verifier đang kiểm |
| `retrying` | `{}` | Verifier trả câu về cho model sửa (không kèm con số bị từ chối) |
| `answer` | như `POST .../messages` | Câu đã qua verifier và đã lưu; event cuối |
| `error` | `{status, detail}` | Lượt lỗi (`502`, `504`, `409`, `500`); event cuối |

`401`, `404`, `503` (không đọc được lịch sử) vẫn trả HTTP status bình thường vì được
kiểm trước khi mở stream. Client ngắt kết nối giữa chừng thì lượt bị huỷ, không lưu gì.

```
curl -N -X POST "http://localhost:8000/sessions/<uid>/messages/stream?profile=deepseek" \
  -H "X-User-Id: <uuid>" -H "Content-Type: application/json" \
  -d '{"question": "Tin mới về HPG?"}'
```

## Luồng một lượt

```
POST /sessions/{uid}/messages
  1. Tải phiên theo (session_uid, user_uid, active); không có -> 404
  2. Tải checkpoint mới nhất (ai_trace.context) và digest các lượt sau nó
  3. Vượt ngân sách -> gấp các lượt trước 3 lượt gần nhất (model rẻ); lỗi -> cắt bớt
  4. User message = <today> + <conversation_history> + <untrusted_input>
  5. run_agent(...); verifier nhận thêm câu hỏi, fact của tóm tắt và các lượt
  6. Tạo digest lượt mới bằng code; lưu checkpoint mới (nếu có) cùng lượt này
  7. Một transaction: turn = MAX(turn) + 1, INSERT user + assistant,
     cập nhật title (nếu chưa có) và updated_at
  8. Trả kết quả
```

## Kiểm chứng (2026-11-18)

Phiên thật trên Node 1, model `deepseek-v4-flash`:

| Lượt | Câu hỏi | Tool | Kết quả |
| --- | --- | --- | --- |
| 1 | Tháng 10/2026 có bao nhiêu bài về HPG? | `count_news` | 229 bài |
| 2 | Còn TCB thì sao? | `count_news` (tự suy ra tháng 10/2026) | 296 bài |
| 3 | Vậy TCB nhiều hơn HPG bao nhiêu phần trăm? | chỉ `calculate`, input lấy từ lịch sử | 29,26%, `verified = true` |

## Compaction (rolling summary)

Theo cơ chế của BOA, với hai điều chỉnh:

| | BOA | Finsights |
| --- | --- | --- |
| Model gấp viết | tóm tắt + danh sách fact (chép lại) | **chỉ** đoạn tóm tắt |
| Fact của lượt đã gấp | model chép, code kiểm từng fact | **code mang theo y nguyên** (số, ngày, link), model không chạm |
| Kiểm đoạn tóm tắt | không | verifier: mọi số/ngày phải có trong lượt được gấp hoặc tóm tắt cũ |
| Checkpoint | cột trong `sessions` | `ai_trace.context` của lượt đã gấp (không đổi schema; bị xoá cùng phiên khi dọn) |

- Model: tier `cheap` của cùng profile (`compactor_model`), prompt `compactor.md`.
- Tóm tắt bị verifier từ chối hoặc dài quá 1.200 ký tự → gửi lại cho model một
  lần kèm phản hồi (không bao giờ cắt cụt, vì lượt mới nhất nằm ở cuối). Vẫn lỗi,
  hoặc provider lỗi/timeout → không gấp, lượt này cắt bớt như cũ.
- Số lượt ("Lượt 2: …") là nguồn hợp lệ cho verifier. Đổi lại, số nguyên nhỏ
  trùng số lượt sẽ không bị bắt.

Kiểm chứng trên Node 1 (2026-11-18), phiên 10 lượt: gấp ở lượt 7 (lượt 1–3), 8,
9, 10, mỗi lần qua verifier ngay lần đầu; tóm tắt giữ ở 600–1.000 ký tự; lượt 10
hỏi lại con số của lượt 1 (đã chỉ còn trong tóm tắt) và trả đúng 229.

## Quota token

Mỗi user có hai cửa sổ quota, theo kiểu Claude:

| Cửa sổ | Mặc định | Setting |
| --- | --- | --- |
| 5 giờ | 150.000 token (khoảng 25 lượt) | `QUOTA_FIVE_HOUR_TOKENS` |
| 7 ngày | 1.500.000 token (khoảng 250 lượt) | `QUOTA_WEEKLY_TOKENS` |

- **Cửa sổ neo, không trượt.** Cửa sổ mở ở lượt đầu tiên sau khi cửa sổ trước đã hết,
  và hết 5 giờ (hay 7 ngày) thì về 0. Đặt setting bằng `0` để tắt loại giới hạn đó.
- **Token tính quota** = input chưa cache + output + 10% token đọc từ prompt cache
  (cache rẻ khoảng 10 lần). Đo trên Node 1 (2026-11-19): câu hỏi đếm khoảng 1.400, liệt
  kê bài 4.700–11.800 (kết quả tool là input mới, không có trong cache), trung bình
  khoảng 6.000.
- **Kiểm tra trước, trừ sau.** Cửa sổ nào đã dùng ≥ hạn mức thì lượt mới bị từ chối với
  `429`, header `Retry-After` (giây), body
  `{"detail": {"message", "limit": "five_hour" | "weekly", "resets_at"}}`. Token chỉ
  biết khi lượt chạy xong, nên lượt cuối (hoặc vài lượt gửi song song) có thể vượt hạn
  mức một chút.
- **Lượt nào cũng bị tính**, kể cả lượt lỗi, timeout hay client ngắt stream giữa chừng:
  token đếm dần trong lúc chạy (`TokenTally`), nên lượt bị huỷ vẫn biết đã tốn bao nhiêu.
  Lượt lỗi giờ cũng lưu `prompt_tokens`/`completion_tokens`.
- Không đọc được quota (DB lỗi) → `503`, agent không chạy. Ghi quota lỗi chỉ log, câu trả
  lời vẫn được trả.
- **Chưa tính** token của compaction (model rẻ, khoảng 1–2k mỗi lần gấp).
- `/runs` không có `X-User-Id` nên không bị giới hạn. Chỉ dùng cho eval: gateway **không
  được** mở endpoint này ra ngoài.

Trạng thái lưu ở bảng `chat_usage` (coredb), một dòng mỗi user: `window_started_at`,
`window_tokens`, `week_started_at`, `week_tokens`. Không tính lại được từ
`chat_messages`, vì điểm bắt đầu của cửa sổ hiện tại phụ thuộc cả chuỗi cửa sổ trước.
Trừ quota là một câu `INSERT … ON DUPLICATE KEY UPDATE`, nên hai lượt đồng thời không làm
mất số liệu của nhau.

`GET /usage`:

```json
{
  "five_hour": {"used": 41250, "limit": 150000, "resets_at": "2026-11-19T17:02:11Z"},
  "weekly": {"used": 412300, "limit": 1500000, "resets_at": "2026-11-23T08:40:05Z"}
}
```

`resets_at = null` khi cửa sổ chưa mở; `limit = null` khi giới hạn đó tắt.

Kiểm chứng trên Node 1 (2026-11-19), với một server tạm đặt `QUOTA_FIVE_HOUR_TOKENS=1000`:
- Sau 2 lượt, `chat_usage` có đúng 1.401 + 11.811 token (khớp công thức với
  `prompt_tokens`, `completion_tokens` và `cache_read_tokens`), hai cửa sổ neo ở lượt 1.
- Lượt tiếp theo nhận `429`, `Retry-After` khoảng 4 giờ 56 phút. Endpoint stream cũng
  `429`; user khác không bị ảnh hưởng.
- Lùi `window_started_at` 6 giờ: cửa sổ 5 giờ về 0, lượt mới mở cửa sổ mới neo ở lượt đó
  (4.714 token), cửa sổ tuần vẫn giữ mốc cũ và cộng dồn (17.926). Lượt sau lại `429`.

## Account và kết nối

| Việc | Account | Node | Quản lý bởi |
| --- | --- | --- | --- |
| Đọc tin (`MYSQL_SOURCE_READONLY_DSN`) | `finsights_ai_ro`, `SELECT` | 2 | coredb, secret `MYSQL_AI_RO_PASSWORD` |
| Lịch sử chat (`MYSQL_CHAT_DSN`) | `finsights_ai_app`, `SELECT/INSERT/UPDATE/DELETE` | 1 | coredb, secret `MYSQL_AI_APP_PASSWORD` |

Cả hai account có `REQUIRE SSL`; service kết nối bằng TLS có verify
(`MYSQL_TLS=true`, CA trong `certs/`). Đổi mật khẩu: cập nhật secret trong
coredb, chạy workflow reconcile, rồi sửa `.env` và khởi động lại server.

## Retention

| Phiên | Xoá hẳn sau |
| --- | --- |
| Người dùng đã xoá (`deleted`) | 30 ngày kể từ lúc xoá |
| Không có lượt mới | 180 ngày kể từ lượt cuối |

```
uv run --env-file .env python -m scripts.purge_chat_history            # dry run
uv run --env-file .env python -m scripts.purge_chat_history --apply    # xoá
```

Chạy mỗi ngày một lần (cron; sau này Dagster). Xoá theo lô 100 phiên, mỗi lô một
transaction; phiên được khoá và kiểm lại điều kiện trước khi xoá, nên phiên vừa
có lượt mới được giữ. Đã kiểm trên Node 1 (2026-11-18): phiên không hoạt động
200 ngày và phiên xoá 40 ngày bị xoá cùng message, phiên xoá 10 ngày được giữ;
chạy lần hai không còn gì để xoá.
