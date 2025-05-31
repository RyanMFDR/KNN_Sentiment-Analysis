# Analisis Sentimen terhadap Kebijakan Program Makan Bergizi Gratis

Anggota :

- Akmal Sabil Amrullah / _Data Scientist_
- Ryan Muhammad Firdaus / _Data Engineer_

Hello Everyone! Nice to share y'all with our projects😍🫶. Pada proyek ini kami berfokuskan dalam menganalisis sentimen terhadap salah satu kebijakan pemerintah yang belakangan ini sedang ramai diperbincangkan yaitu **"Makan Bergizi Gratis"**

🔗 Sumber Data : https://x.com/ (twitter)

🔗 File Laporan Akhir : https://docs.google.com/document/d/1vYcnNznTUEQmMFBdb_ZF5qvC2ChoEotYT4Kfsdjn-JQ/edit?tab=t.0

## Cara Menjalankan Kode

1. Install dependencies yang dibutuhkan:

```bash
pip install -r requirements.txt
```

2. Jalankan preprocessing data:

```bash
python scripts/preprocess.py
```

3. Buka notebook analisis untuk menjalankan analisis dan visualisasi

```bash
analisis.ipynb
```

## Direktori Projek

```bash
/
├── scripts/
│ └── preprocess.py #script preprocess
├── analisis.ipynb #analisis dan visualisasi
├── requirements.txt #dependencies Python
└── README.md
```

## Scraping Data

- Data dikumpulkan secara manual dari Twitter (https://x.com/) dengan kata kunci "Makan Bergizi Gratis" atau "MBG"

- Tweet-tweet yang relevan disalin dan disimpan secara manual dalam format CSV untuk kemudian diolah lebih lanjut.

- Setelah data terkumpul, dilakukan pembersihan dan persiapan data sebelum proses analisis sentimen.
