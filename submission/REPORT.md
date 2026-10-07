# Lab 21 — Evaluation Report

**Họ tên**: Lục Tiến Đạt  **MSSV**: 2A202602969  **Ngày**: 2026-10-08
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42, tỉ lệ 90/10) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json, suggested_max_length=256; tier T4 cấu hình trần 1024 để đảm bảo không bị cắt đuôi generation)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 max_steps |

**Template có giữ khối `<think>` không?** `Có` — Theo kiểm tra tự động tại `results/template_check.json`, tokenizer của Qwen3.5 giữ nguyên cặp thẻ `<think> ... </think>` và phần thân suy luận (`open_tag_present: true`, `body_present: true`, kết luận: *"reasoning preserved — safe to train on traces"*). Vì thế hệ thống không bị lỗi nuốt tag suy luận và không cần can thiệp chỉnh sửa Jinja template thủ công.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng token nằm trong vùng tính loss) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3172.5 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1037.4 |
| (c) LoRA fine-tune | 0.965 | 0.544 | 1.000 | 1363.4 |

**(b) có thật sự mạnh hơn (a) không?** `Có`. Baseline (b) cải thiện vượt bậc so với (a): target accuracy từ 0.000 tăng lên 0.765, tỷ lệ format JSON hợp lệ từ 0.000 đạt tuyệt đối 1.000, đồng thời latency trung bình giảm từ 3172.5 ms xuống 1037.4 ms nhờ prompt tối ưu cắt bỏ các câu chào hỏi dài dòng của base model.
Bạn có sửa `OPTIMIZED_PROMPT` không? `Không sửa` — Giữ nguyên 100% prompt chuẩn có SHA mã hóa `719e74d3b6232053` như repo cung cấp, nhằm đảm bảo tính khách quan và liêm chính tuyệt đối của phép so sánh khoa học.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6265 | 0.965 | 391.8 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5378 | 0.970 | 270.2 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.000 | 402.7 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 471.1 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trích xuất xếp hạng theo cột **target (NB5 §4)**: `attn_only` (0.970) > `correct` (0.965) > `qlora` (0.940) > `wrong_lr` (0.000).

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Run `attn_only` được khớp ngân sách tham số qua `matched_rank()` đạt 32,456,704 tham số (lệch đúng 0.025% so với `correct`, hoàn toàn thỏa mãn tiêu chí kiểm định <5%). Trên tập target, `attn_only` đạt độ chính xác 0.970, nhỉnh nhẹ hơn một mẫu so với `correct` (0.965), và thứ tự này đồng điệu với train loss khi `attn_only` đạt 0.5378 so với 0.6265 của `correct`. Hiện tượng này cho thấy khi dồn toàn bộ dung lượng tham số vào một nhóm module hẹp với rank cực kỳ lớn ($r=283$), adapter có khả năng học thuộc và ghi nhớ mẫu hình chuyên biệt của tập dữ liệu hẹp rất sâu. Tuy nhiên, việc tăng rank tới 283 tạo ra chi phí tính toán và lưu trữ ma trận LoRA cồng kềnh hơn nhiều lúc phục vụ. Ngược lại, chiến lược gắn adapter vào toàn bộ các lớp tuyến tính (`all-linear`) với rank nhỏ ($r=16$) phân bổ đều tri thức học được trên toàn bộ mạng lưới, mang lại sự ổn định và cân bằng kiến trúc tốt hơn nhiều cho các kịch bản tổng quát.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` (sử dụng learning rate $1\times 10^{-5}$ của full fine-tuning) gần như đi ngang và kẹt cứng ở mức 1.5702, hoàn toàn không hội tụ so với mức giảm sâu 0.6265 của `correct`. Hậu quả trên tập kiểm thử là một sự sụp đổ toàn diện: độ chính xác target tụt về 0.000 và tỷ lệ format đúng là 0.000 do mô hình không đủ biên độ cập nhật trọng số để nắm bắt cấu trúc JSON. Nếu một kỹ sư chỉ quan sát loss phẳng và kết quả suy luận tồi tệ mà không đối chiếu siêu tham số LR, họ sẽ dễ dàng ngộ nhận rằng LoRA hoạt động kém hiệu quả, nghi ngờ tập dữ liệu bị gán nhãn sai, hoặc vội vã kết luận mô hình Qwen3.5-4B không có khả năng học tác vụ phân loại tiếng Việt. Trong thực tế, LoRA đòi hỏi mức learning rate lớn hơn khoảng $10\times$ so với full fine-tuning (thang $10^{-4}$) để bù đắp cho việc các ma trận tích chập adapter được khởi tạo ngẫu nhiên từ 0 và bị giới hạn nghiêm ngặt trong không gian rank thấp.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Run `qlora` đã chứng minh ưu thế vượt trội về mặt tài nguyên khi cắt giảm đỉnh VRAM từ 8.78 GB xuống chỉ còn 3.86 GB, giải phóng tới hơn 4.92 GB VRAM (~56% dung lượng bộ nhớ). Tuy nhiên, sự đánh đổi là rất rõ ràng: thời gian huấn luyện bị kéo dài thêm gần 20% (từ 391.8 s lên 471.1 s) do độ trễ khử lượng tử hóa (dequantization) liên tục giữa 4-bit và 16-bit trong từng bước lan truyền, đồng thời điểm target accuracy bị tụt từ 0.965 xuống 0.940. Những số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị của nhà phát triển mô hình năm 2026: trên dòng Qwen3.5, sai số làm tròn của lượng tử hóa 4-bit NF4 gây suy hao đáng kể tới khả năng biểu diễn ngữ nghĩa tinh tế. Khi GPU khả dụng (như T4 16GB) đã hoàn toàn đủ sức chứa mô hình 16-bit với LoRA ở mức tiêu thụ 8.78 GB, việc đánh đổi độ chính xác và tốc độ huấn luyện để lấy thêm VRAM nhàn rỗi là một quyết định kỹ thuật không tối ưu.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.200` · `regression Δ = -0.247` · `valid_trace_rate = 0.0`

Diễn giải:
Cổng hồi quy tại `results/verdict.json` chính thức trả về kết quả `FAILED` bởi vì chỉ số suy giảm năng lực tổng quát `regression Δ = -0.247` đã vượt qua ngưỡng dung sai cho phép là 0.020. Trong khi tác vụ chuyên biệt CSKH tăng trưởng ngoạn mục từ 0.765 lên 0.965 (tăng +20.0 điểm %), khả năng trả lời các câu hỏi chỉ dẫn và kiến thức tiếng Việt phổ thông của mô hình lại sụt giảm nghiêm trọng từ 0.7911 xuống 0.5444. 

Kết quả này là một minh chứng thực nghiệm kinh điển cho hiện tượng Catastrophic Forgetting (quên lãng thảm họa) trong huấn luyện thích ứng mô hình ngôn ngữ. Khi chúng ta fine-tune mô hình chỉ trên 250 mẫu dữ liệu thuần túy định dạng JSON ticket CSKH với cường độ gradient mạnh, không gian biểu diễn trọng số bị bẻ cong hoàn toàn về phía cấu trúc đầu ra đặc thù, làm đứt gãy các liên kết tri thức nền tảng sẵn có của base model. Điều này khẳng định rằng dù bản fine-tune đạt điểm nghiệp vụ rất cao, ta không thể đưa nó vào vận hành như một trợ lý đa năng. Để vượt qua cổng hồi quy này theo đúng bài học tại slide deck §6.3, giải pháp kỹ thuật bắt buộc là phải phối trộn thêm từ 1% đến 5% dữ liệu hồi quy (replay buffer từ tập instruction tổng quát) trong quá trình huấn luyện nhằm duy trì tri thức phổ quát cho mô hình.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper không gọi. Hỏi cho biết thôi. Shop hỗ trợ tốt. (Ticket 48) | `{"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"}` | Nhầm intent sang hỏi thông tin, sai urgency. | `{"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"}` | ✅ **FT thắng**: Bắt đúng 100% cả 4 trường, trích xuất chính xác tên sản phẩm và mức độ khẩn cấp thấp. |
| 2 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. Mong shop phản hồi. Nhờ shop kiểm tra. (Ticket 49) | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | Trích xuất sai sắc thái sentiment do hiểu nhầm câu nhờ kiểm tra. | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | ✅ **FT thắng**: Fine-tune phân loại chính xác ý định hỏi giá và cảm xúc trung tính chuẩn xác. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. (Ticket 3) | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | Phân loại đúng toàn bộ cả 4 trường (`urgency: thap`, `sentiment: tich_cuc`). | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | ❌ **FT thua**: Mô hình FT bị overfit vào từ khóa hoàn tiền "Chưa thấy tiền" nên tự động nâng mức độ khẩn cấp lên `trung_binh`, bỏ qua cụm từ "Khi nào tiện". |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. (Ticket 5) | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | Bắt đúng cụm từ giảm nhẹ "Khi nào tiện" để gán `urgency: thap`. | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | ❌ **FT thua**: FT đánh giá sự cố thiếu phụ kiện là mặc định mức độ trung bình, mất đi sự nhạy cảm với sắc thái khách hàng không vội. |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. (Ticket 12) | `{"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | Nhận diện đúng mức độ khẩn cấp thấp từ chỉ dẫn in-context. | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | ❌ **FT thua**: Lặp lại thiên kiến gán nhãn `urgency: trung_binh` cho mọi khiếu nại sản phẩm lỗi, thất bại trong việc ghi nhận mức độ `thap`. |

**Có mẫu chung nào ở các ca FT thua không?**
Có một mẫu hình lỗi sai cực kỳ nhất quán giữa các ca fine-tune thua: Mô hình bị mất tính linh hoạt ngữ cảnh ở trường **`urgency`**. Trong tập huấn luyện, đa số các ticket khiếu nại (hoàn tiền, sản phẩm lỗi, thiếu phụ kiện) đều có mức độ khẩn cấp từ trung bình đến cao. Do đó mô hình fine-tune đã hình thành một mối tương quan giả định (spurious correlation) rằng hễ có sự cố là `urgency` tối thiểu phải là "trung_binh", hoàn toàn bỏ qua các tín hiệu giảm nhẹ tinh tế trong câu như "Khi nào tiện" hay "Hỏi cho biết thôi". Trong khi đó, baseline prompt (b) nhờ có hướng dẫn luật lệ tường minh trong system prompt lại xử lý các trường hợp biên này chuẩn xác hơn.

---

## 7. Kết luận & điều tôi học được

**Kết luận.**
Bản fine-tune LoRA `correct` trong bài thí nghiệm này đạt bước nhảy vọt đáng kể trên bài toán mục tiêu, nâng độ chính xác phân loại ticket CSKH tiếng Việt từ 76.5% (prompt tối ưu) lên 96.5% (+20.0%), đồng thời bảo toàn định dạng JSON chuẩn xác 100%. Tuy nhiên, câu trả lời dứt khoát cho câu hỏi **"Có nên deploy bản fine-tune này lên môi trường production hay không?"** là: **CHƯA ĐƯỢC PHÉP DEPLOY TRỰC TIẾP** cho một hệ thống tổng quát. Nguyên nhân cốt lõi là cổng hồi quy đã đánh trượt mô hình do sự sụt giảm nghiêm trọng 24.7 điểm % ở năng lực xử lý ngôn ngữ phổ thông (catastrophic forgetting). Mô hình hiện tại chỉ có thể được triển khai an toàn nếu đặt sau một Router kiến trúc Microservice chuyên biệt (chỉ nhận đúng các request là ticket CSKH đã được lọc).

Đòn bẩy thực sự chi phối chất lượng trong toàn bộ pipeline này chính là **Learning Rate** và **Cơ chế Masking kết hợp với tính Liêm chính của Dữ liệu**, chứ không nằm ở việc cố gắng nâng rank tham số lên cao. Thực nghiệm ở NB4 đã chứng minh rằng sai lệch thang LR sẽ hủy diệt hoàn toàn mô hình (về 0%), trong khi việc tăng rank từ 16 lên 283 ở `attn_only` chỉ mang lại cải thiện cận biên không đáng kể nhưng làm phình to chi phí phục vụ. Fine-tuning không phải là một chiếc đũa thần giải quyết mọi vấn đề; nó là một bài toán đánh đổi khắt khe giữa sự thích ứng chuyên sâu và tính bảo toàn tri thức, đòi hỏi kỹ sư phải luôn kiểm chứng qua các cổng đo lường khách quan thay vì tin tưởng mù quáng vào train loss.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Sự khác biệt bản chất giữa thang Learning Rate của Full Fine-Tuning ($10^{-5}$) và LoRA ($10^{-4}$)**: Do các ma trận adapter $A$ và $B$ khởi tạo trong không gian chiếu rank thấp và xuất phát điểm là cập nhật rỗng, chúng cần một tốc độ học lớn hơn gấp 10 lần để thích nghi hiệu quả; áp dụng mù quáng LR của full-FT sẽ khiến loss đi ngang và mô hình hoàn toàn bất động.
2. **Loss Masking và Token Alignment là tuyến phòng thủ số một**: Việc giải mã ngược các token trong `mask_proof.json` để chứng minh chỉ có lượt trả lời của Assistant chịu phạt loss là điều kiện tiên quyết sống còn. Nếu để lọt prompt vào loss mask, mô hình sẽ học thói quen viết lại câu hỏi thay vì giải quyết bài toán.
3. **Ảo tưởng về Train Loss và bài học về Cổng Hồi Quy Đa Nhóm**: Xếp hạng chất lượng mô hình bằng train loss là một sai lầm chết người trong thực tế. Đánh giá LLM bắt buộc phải đo lường trên 4 trục độc lập: Target, Regression, Format, Latency. Nhận diện được ca fine-tune thất bại ở cổng hồi quy mang lại giá trị kỹ thuật cao hơn nhiều so với việc cố tình nới lỏng tiêu chí để lấy kết quả đỗ giả tạo.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ triển khai giải pháp bổ sung từ 3% đến 5% dữ liệu hồi quy (Replay Buffer trích xuất từ tập chỉ dẫn tiếng Việt tổng quát hoặc Wikipedia tiếng Việt) vào tập huấn luyện của NB3. Mục tiêu là kiểm chứng giả thuyết: liệu việc neo giữ trọng số thông qua replay data có giúp mô hình vượt qua Cổng Hồi Quy (`regression Δ > -0.02`) để đạt trạng thái PASSED trọn vẹn mà vẫn duy trì được độ chính xác phân loại ticket CSKH trên 95% hay không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
