# Laporan Proyek Machine Learning - Muhammad Revin Arnan

## Domain Proyek

Dalam era informasi digital saat ini, berita dan informasi tersebar dengan cepat melalui *platform* *online* dan media sosial. Sayangnya, bersama dengan peningkatan aksesibilitas informasi, masalah berita *hoax* atau palsu juga semakin merajalela. Berita *hoax* dapat mempengaruhi persepsi masyarakat, menciptakan kebingungan, dan bahkan menyebabkan konsekuensi serius, termasuk kerusuhan sosial, dan ketidakstabilan politik [4]. 

Saat ini di Indonesia sedang memasuki masa pesta demokrasi. Tidak jarang yang terlihat pada pesta demokrasi sebelum-sebelumnya, banyak oknum atau *buzzer* yang sengaja menyebarkan berita *hoax* tentang lawan politiknya. Masyarakat perlu pintar-pintar dalam menyaring berita yang tersebar di media sosial, supaya tidak termakan 'taktik' busuk dari penyebar berita tersebut.

Untuk mengatasi masalah ini, proyek *machine learning* tentang klasifikasi berita *hoax* dapat membantu dalam mengidentifikasi dan membedakan antara berita yang sah dan berita *hoax*. Dengan menggunakan teknik *Natural Language Processing*, model dapat dilatih untuk secara otomatis menganalisis teks berita dan memberikan prediksi apakah berita tersebut dapat dipercaya atau tidak.

## Business Understanding

### Problem Statements

Bagaimana cara menentukan berita tersebut adalah berita yang sah atau berita yang *hoax* kepada pengguna?

### Goals

Pengguna dapat mengetahui berita yang berjudul A dan isi berita A' termasuk ke dalam berita sah atau *hoax*, dengan menggunakan model *machine learning* dan *dataset* yang sesuai.

### Solution statements

Membuat *deep learning model* dengan menggunakan algoritma *Recurrent Neural Network* (RNN) dan beberapa *layer* lainnya yang akan dijelaskan pada bagian "Modeling".

## Data Understanding

*Dataset* yang digunakan pada proyek *machine learning* ini adalah data "*Fake and real news dataset*" dari Kaggle. *Dataset* dapat diakses pada *link* berikut: [Link dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).  

### Variabel-variabel pada *Fake and Real News* dataset adalah sebagai berikut:

- *title*: merupakan judul dari suatu berita yang ada.
- *text*: merupakan *raw* teks dari suatu berita tertentu.
- *subject*: adalah jenis golongan berita tersebut.
- *date*: adalah tanggal terbit suatu berita.

Tahapan pemahaman *dataset* yang dilakukan diantaranya:

1.  Melihat macam dan jumlah kolom dengan df.info,

    <img width="185" alt="df_info" src="https://github.com/revinarnan/sub-1-ml-terapan/assets/45119832/eaabc97d-6dc9-4c81-ad1a-11b397a503b2">
    
    *Gambar 1. Dataframe Info*
    
    Pada Gambar 1, dapat dilihat bahwa *dataset* memiliki 5 kolom dan 44898 baris.

2. Memvisualisasikan jumlah dari masing-masing subjek pada berita,
   <img width="629" alt="subject_type_vis" src="https://github.com/revinarnan/sub-1-ml-terapan/assets/45119832/cf0f4703-a6fc-4e27-92e8-5c2ba4be3d08">
   
   *Gambar 2. Grafik Persebaran Subjek Berita*

   Dari hasil *diagram plot*, dapat dilihat banyaknya data dari masing-masing *subject* berita. Subjek berita paling banyak adalah politik, dan paling sedikit adalah subjek *middle-east.* Sebagai catatan, dapat dilihat juga terdapat subjek dengan maksud yang sama, yaitu politik, yang memiliki dua batang *chart* berbeda. Hal ini dikarenakan, sebagian batang termasuk ke dalam berita *fake* dan satu lainnya termasuk ke dalam kategori *real*. 

3. Melihat kata yang sering muncul dengan *wordcloud*,

   <img width="191" alt="word_cloud_vis" src="https://github.com/revinarnan/sub-1-ml-terapan/assets/45119832/118fa305-56df-4cba-9e2f-f8b4244d8b3c">

   *Gambar 3. Word Cloud*

   Kata yang sering muncul dalam *dictionary* adalah Donald Trump, US, White House, dan sebagainya.

4. Melihat perbandingan data *Fake* dan data *Real* *News*,

   <img width="436" alt="target_dis" src="https://github.com/revinarnan/sub-1-ml-terapan/assets/45119832/f0f282db-a6c6-4329-b228-a3ad70465018">

   *Gambar 4. Diagram Batang Label*

   Pada Gambar 4, jumlah data *fake* dan data *real* cukup seimbang, tidak terlalu jauh selisih banyaknya data.

5. Melakukan pengecekan terhadap *null values*.

   <img width="100" alt="df_na" src="https://github.com/revinarnan/sub-1-ml-terapan/assets/45119832/b852d0cf-1814-49b5-8133-389732e7fbef">

   *Gambar 5. Hasil Cek Null Values*

   Dataset tidak memiliki data yang *null*, sehingga bisa langsung dilanjutkan ke proses berikutnya.

## Data Preparation

Proses pembersihan data dan preparasi yang dilakukan diantaranya sebagai berikut:

- Menggabungkan kolom '*title*' dan '*text*' menjadi satu sebagai *features*.
- *Drop* kolom '*title*', '*subject*', dan '*date*' karena tidak dibutuhkan dalam proses latih.
- Menghilangkan *stopwords* Bahasa Inggris, karena *stopwords* tidak memiliki arti yang penting dalam kalimat. *Stopwords* dapat dihilangkan dan tidak akan berpengaruh terhadap model yang dibangun. Contoh *stopwords* Bahasa Inggris adalah *the, we, have*, dll.
- Mengkonversi teks menjadi huruf kecil semua, supaya teks memiliki struktur yang seragam
- Membersihkan teks dengan menggunakan *regular expression*:
  - Menghilangkan teks url dengan substitusi.
  - Menghilangkan karakter selain teks dengan substitusi.
- Membagi data latih dan data training dengan test dengan train_test_split, dengan komposisi 70% sebagai latih, dan 30% sebagai uji.
- *Tokenizing* *text*: Menandai setiap kata dengan angka dan memetakan data *text* pada *token* tersebut.
- Membatasi setiap teks latih dengan maksimal 300 kata setiap data. Jika lebih akan dipotong, dan jika kurang akan ditambahkan padding.

## Modeling

Proyek ini menggunakan model *deep learing* dengan *Recurrent Neural Network* (RNN) sebagai algoritmanya. RNN bekerja dengan cara mengolah input baru dan memprosesnya dengan berbagai informasi yang telah didapatkan sebelumnya. Informasi-informasi ini diingat di dalam memori internal milik RNN. Input baru akan diproses melalui sebuah *loop* yang mengandung beberapa informasi sebelumnya. Karena itulah, RNN tidak hanya mempertimbangkan input baru itu saja, namun juga melibatkan informasi yang telah didapatkan sebelumnya [1], [3].

Pada model *deep learning* yang dibangun, terdapat beberapa layer, diantaranya:

- *Embedding layer*: dengan dimensi input sebanyak 10000, dan dimensi output sebanyak 128.
- *Bidirectional layer* 1: menggunakan layer LSTM dengan parameter 'return_sequence' bernilai *True*. Artinya, pada layer ini, LSTM akan menghasilkan *output* pada setiap *timestamp*-nya. Hal ini digunakan karena kita perlu mengetahui urutan kata pada setiap data teks yang dianalisis.
- *Bidirectional layer* 2: menggunakan layer LSTM dengan *weight* = 16. Parameter 'return_sequence' pada layer ini bernilai False, sehingga LSTM hanya akan mengembalikan output pada *timestamp* terakhir.
- *Dense layer* 1: memiliki bobot bernilai 32 dengan fungsi aktivasi *Rectified Linear Unit* (ReLU). 
- *Dropout layer*: untuk mencegah *overfitting*, memiliki nilai bobot 0,5.
- *Dense layer* 2: bisa juga disebut dengan *output layer* atau *layer* terakhir, menggunakan fungsi aktivasi Sigmoid.

Selain itu, model juga menggunakan *Adam Optimizer* dengan *learning rate* bernilai 0,01 dengan *binary_crossentopry loss function*.

## Evaluation

Metriks yang digunakan pada proyek ini adalah metriks *accuracy*. Metriks ini digunakan untuk mengukur performa model. Metrik ini mengukur sejauh mana model dapat mengklasifikasikan data dengan benar. Secara sederhana, *accuracy* menghitung persentase prediksi yang benar dari total jumlah data yang dievaluasi [2]. Rumus metriks *accuracy* adalah sebagai berikut:
$$
Accuracy = {TN + TP \over TN + FP + TP + FN}
$$
Sederhananya, metriks ini menghitung jumlah persentase prediksi benar dari seluruh hasil prediksi. Hasil metriks ini dari model yang dikembangkan adalah sebagai berikut: 

<img width="291" alt="model_acc" src="https://github.com/revinarnan/sub-1-ml-terapan/assets/45119832/08618764-3129-4f11-b7bc-30def87a504d">

*Gambar 6. Grafik Akurasi Model*

<img width="291" alt="model_loss" src="https://github.com/revinarnan/sub-1-ml-terapan/assets/45119832/e3da8aaa-8d3e-4a79-bcd2-dc553e00ed68">

*Gambar 7. Grafik Loss Model*

<img width="401" alt="model_ev" src="https://github.com/revinarnan/sub-1-ml-terapan/assets/45119832/62a2622f-f4cc-4fa7-87f3-b946bcead712">

*Gambar 8. Hasil Evaluasi Model*

Model mendapatkan nilai akurasi sebesar 98,5% dengan loss model sebesar 7%.

Ditampilkan juga hasil *confusion matrix* dari hasil prediksi sebagai berikut:

<img width="236" alt="conf_matrix" src="https://github.com/revinarnan/sub-1-ml-terapan/assets/45119832/5f0ea859-47b3-4c15-922a-f7e5437f78f4">

*Gambar 9. Hasil Confusion Matrix Model*

Pemetaan tabel dari hasil *confusion matrix* pada gambar di atas.

*Tabel 1. Pemetaan Confusion Matrix*

|                | *Fake*                    | *Original*                |
| -------------- | ------------------------- | ------------------------- |
| ***Fake***     | ***True Negative (TN)***  | ***False Positive (FP)*** |
| ***Original*** | ***False Negative (FN)*** | ***True Positive (TP)***  |

Dari *confusion matrix* di atas, dapat  dilakukan analisis bahwa model dapat memprediksi 7124 *predicted label* 'Fake' dari 7177 *actual label* 'Fake'. Model juga dapat memprediksi sebanyak 6141 *predicted label* 'Original' dari 6293 *actual label* 'Original'. Selain itu, sebanyak 53 label diprediksi sebagai 'Original' yang seharusnya berlabel 'Fake', dan sebanyak 152 label diprediksi model sebagai 'Fake' yang seharusnya berlabel 'Original'.

## Kesimpulan

Model *machine learning* dapat dilatih untuk mengklasifikasikan berita palsu dan berita sesungguhnya. Dengan menggunakan teknik *deep learning*, model dapat memberikan akurasi sebesar 98,5%. Dalam kasus ini, hasil yang didapatkan cukup baik. Namun, model menggunakan *dataset* kumpulan berita dari sumber berbahasa Inggris, sehingga jika ingin digunakan dalam kasus klasifikasi berita dalam negeri, model perlu menggunakan *dataset* berita berbahasa Indonesia. Tentunya dengan beberapa penyesuaian pada tahap preparasi data.

## Saran

Untuk dapat memprediksi berita *hoax* di Indonesia, model perlu dilatih dengan menggunakan *dataset* dengan Bahasa Indonesia.

## Daftar Pustaka

[1] S. Kostadinov, “How recurrent neural networks work,” Medium, https://towardsdatascience.com/learn-how-recurrent-neural-networks-work-84e975feaaf7 (accessed Jun. 29, 2023). 

[2] B. Harikrishnan N, "Confusion Matrix, Accuracy, Precision, Recall, F1 Score", Analytics Vidhya, https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd (accessed Jun. 25, 2023).

[3] A. P. Gema and D. Suhartono, “Recurrent neural network (RNN) Dan Gated Recurrent Unit (GRU),” Recurrent Neural Network (RNN) dan Gated Recurrent Unit (GRU), https://socs.binus.ac.id/2017/02/13/rnn-dan-gru/ (accessed Jun. 25, 2023). 

[4] K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, “Fake news detection on social media: A Data Mining Perspective: ACM SIGKDD Explorations Newsletter: Vol 19, no 1,” ACM SIGKDD Explorations Newsletter, https://dl.acm.org/doi/10.1145/3137597.3137600 (accessed Jun. 24, 2023). 
