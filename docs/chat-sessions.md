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

Mã lỗi của `POST .../messages`: `401` thiếu/sai user, `404` không thấy phiên,
`409` lượt bị ghi trùng, `502`/`504` agent lỗi (giống `/runs`), `503` DB lịch sử
không đọc được (agent không chạy). Nếu chỉ bước **lưu** lỗi, câu trả lời vẫn được
trả với `turn = null`.

`POST /runs` giữ nguyên (không lưu gì) cho eval và demo.

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
