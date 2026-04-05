Nhận xét jupytext dữ liệu ảnh (eda)
Code đang để nhận xét class imbalance lên trước pixel và làm như một phần về mô tả data trong khi yêu cầu không có như thế, và đang sai thứ tự vị trí
Code vẽ box plot brightness vẫn đang sort làm cái label rất là khó đọc vì nó lộn xộn, đừng có sort (fix cái axis label dùm)
chỗ scatter 45 lớp, tăng font size lên một xíu
Output cell và nhận xét chưa consistent về số, xí bạn fix hết lỗi rồi fix cái nhận xét đánh giá này sau
Nhận xét jupytext dữ liệu ảnh (processing)
Chưa có pip install skimage hay sao đó, tách một file requirements.txt để cài đặt full thư viện vì tôi sẽ chạy lỗi 
Nhận xét jupytext dữ liệu ảnh (advanced)
resize ảnh phải chọn lại size dựa trên notebook trước, tự nhiên chọn 64x64, không rõ lắm, rồi sample size gì có 50 vậy, chiếu toàn bộ không được hả
tiếp theo là PCA components, sao lại là 100, tưởng đề bài yêu cầu là hmm 50?, tôi không nhớ rõ var n_components là gì trong PCA nhưng chek lại yêu cầu đề bài, hiện số component cho 90, 95, 99% thì phải hiện số thực sự chứ đừng ghi chung chung >100, tôi nhớ có note lỗi này rồi
đọan eigen images ổn nhưng sao không visualize hết, có vấn đề gì hả
tiếp theo là 2d projecttion gì đó, tự nhiên chỉ chọn 10 lớp, rất là kỳ, vì dataset có 45 lớp hay sao mà?, chưa có 3D PCA với T-SNE hay sao đó
Không hiểu kiểu gì, rất nhiều notes về việc tham số chính không phải kernel size, không phải gaussian blur mà là Threshold, thiếu thuật toán sai tùm lum, không hiểu intend đề bài, ĐỌC LẠI PDF, không có đọc cái file md rồi làm lại khúc này
Nhận xét jupytext tabular data
Có một điều buồn cười là DBSCAN ít làm thay dổi phân phối dữ liệu nhất, trong khi IQR lại làm thay đổi nhiều nhất, lý do gì để mình chọn một phương pháp nào đó chưa có nêu ra một cách tường minh
code heuristic và làm đơn giản rất nhiều, ví dụ như chỉ lấy subsample hmm
check thật kỹ ý d) và e) trong pdf, thấy jupytext bất ổn quá
Nhận xét juputext text data
thế quái nào vẫn thiếu plot lên cái số ký tự (số từ, số câu thì có nhưng số ký tụ thì không)


[NEW FEEDBACK]

Jupytext 01 ----
Phần pixel: kiểm định với 5 lớp đại diện ở phần pixel, hơi awkard, why 5 lớp này, không nói rõ, kiểm định hơi kỳ hmm

Phần class imbalance: Vẽ cái pie chart thay vì bar chart như hiện tại

Phần chọn kích thước resize, có lẽ nên chọn 128x128 vì nó có KNN cao và độ mất mát thông tin ở mức trung bình

Phần chuyển đổi không gian màu nên sài ảnh 128x128 như được chọn lựa ở trên, kết quả thay đổi dẫn đến nhận xét, phân tích kêt luận không còn chính xác

Phần chuẩn hóa có thể bỏ cell mà visualize cái ảnh trước và sau chuẩn hóa

Phần Sobel Canny gì đó chưa chỉnh gì hết, làm sai hết rồi, haiz xóa code đọc lại đề rồi làm (chú ý ngưỡng mới là siêu tham số chính)

