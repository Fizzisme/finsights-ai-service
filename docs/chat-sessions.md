# Chat sessions và lịch sử hội thoại

Trạng thái: **giai đoạn 1 đã triển khai** (nhiều lượt, lưu lịch sử, cửa sổ trượt).
Compaction bằng LLM để sang giai đoạn 2.

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
2. **Cửa sổ trượt.** Lượt mới đọc digest của `KEEP_RECENT_TURNS = 3` lượt gần nhất.
3. **Ngân sách cố định.** Lịch sử đưa vào model tối đa `HISTORY_TOKENS = 2000`
   (ước lượng 3 ký tự/token vì tiếng Việt tốn token). Vượt ngân sách thì bỏ digest
   cũ nhất, luôn giữ lượt mới nhất; lượt mới không bao giờ fail vì lịch sử dài.
4. **Fact cũ là nguồn hợp lệ.** Câu hỏi và fact của các lượt trong cửa sổ được
   verifier chấp nhận như tool result (kể cả làm input cho `calculate`). Chỉ lượt
   có `verified = true` mới để lại fact, nên con số chưa kiểm chứng không lan sang
   lượt sau.
5. **Lịch sử là dữ liệu, không phải lệnh.** Nó chứa chữ người dùng gõ và nội dung
   lấy từ bài báo, nên được bọc trong `<conversation_history>` (không thể tự đóng
   thẻ) và prompt coi nó như `<untrusted_input>`.

Code: `app/graph/history.py` (digest, cửa sổ, block, nguồn cho verifier).

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
| `ai_trace` (JSON) | Dòng assistant: `digest`, `tool_calls`, `verified`, `unsupported`, `latency_ms` |
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
  2. Tải digest 3 lượt gần nhất -> cửa sổ (cắt theo ngân sách)
  3. User message = <today> + <conversation_history> + <untrusted_input>
  4. run_agent(...); verifier nhận thêm câu hỏi và fact của cửa sổ
  5. Tạo digest lượt mới bằng code
  6. Một transaction: turn = MAX(turn) + 1, INSERT user + assistant,
     cập nhật title (nếu chưa có) và updated_at
  7. Trả kết quả
```

## Kiểm chứng (2026-11-18)

Phiên thật trên Node 1, model `deepseek-v4-flash`:

| Lượt | Câu hỏi | Tool | Kết quả |
| --- | --- | --- | --- |
| 1 | Tháng 10/2026 có bao nhiêu bài về HPG? | `count_news` | 229 bài |
| 2 | Còn TCB thì sao? | `count_news` (tự suy ra tháng 10/2026) | 296 bài |
| 3 | Vậy TCB nhiều hơn HPG bao nhiêu phần trăm? | chỉ `calculate`, input lấy từ lịch sử | 29,26%, `verified = true` |

## Giai đoạn 2: compaction bằng LLM

Khi có dữ liệu thật cho thấy phiên dài bị mất ngữ cảnh do cắt bớt: các lượt rơi
khỏi cửa sổ được một model rẻ gấp thành rolling summary, lưu làm checkpoint
(cần thêm cột qua coredb). Code kiểm mọi fact model giữ lại phải khớp y nguyên
với digest gốc (như `facts_verifier` của BOA); gấp lỗi thì quay về cắt bớt.

## Còn mở

1. **Account chat:** chuyển `MYSQL_CHAT_DSN` sang `finsights_ai_app` (do coredb
   quản lý, `REQUIRE SSL`) rồi bỏ `finsights_ai_rw` tạo tay. Service đã kết nối
   MySQL bằng TLS có verify (`MYSQL_TLS=true`, CA trong `certs/`).
2. **`finsights_ai_ro` ở Node 2:** khai báo trong coredb để không bị xoá khi
   reconcile; cần secret `MYSQL_AI_RO_PASSWORD` trước khi merge.
3. Thời gian giữ lịch sử (retention).
