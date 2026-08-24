<p align="center">
  <img src="assets/realesrgan_logo.png" height=120>
</p>

<div align="center">

# HG-ESR-NET

**Mesin Inferensi Super-Resolution dengan Dukungan Arsitektur Universal**

[English](README.md) | [简体中文](README_CN.md) | [Bahasa Indonesia](README_ID.md)

[![License](https://img.shields.io/github/license/Hanzet22/HG-ESR-NET.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](#-dependensi-dan-instalasi)
[![Version](https://img.shields.io/badge/version-1.2.0-orange.svg)](VERSION.md)
[![Open issue](https://img.shields.io/github/issues/Hanzet22/HG-ESR-NET)](https://github.com/Hanzet22/HG-ESR-NET/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/Hanzet22/HG-ESR-NET)](https://github.com/Hanzet22/HG-ESR-NET/issues)

</div>

**HG-ESR-NET** adalah mesin inferensi super-resolution universal, dibangun di
atas pipeline upscaling Real-ESRGAN yang sudah terbukti, dan diperluas dengan
lapisan deteksi arsitektur hybrid yang secara otomatis memuat **25+ arsitektur
SR/restorasi** dari satu file checkpoint saja — tanpa perlu menulis kode arch
secara manual.

Cukup arahkan ke file model `.pth` / `.safetensors` / `.ckpt` apapun dari
OpenModelDB, HuggingFace, atau hasil training sendiri, dan engine ini akan
mendeteksi arsitekturnya, membangun network-nya, lalu menjalankan inferensi —
caranya sama persis untuk ESRGAN, SPAN, OmniSR, SwinIR, HAT, DAT, RealCUGAN,
dan semua arsitektur lain yang didukungnya.

---

## ✨ Yang Ditambahkan HG-ESR-NET

- **Universal model loader** (`realesrgan/universal_loader.py`) — deteksi
  hybrid: mencoba [Spandrel](https://github.com/chaiNNer-org/spandrel)
  terlebih dahulu (25+ arsitektur, hyperparameter otomatis terdeteksi
  langsung dari state_dict mentah), lalu fallback ke registry `archs2/`
  yang ditulis manual (RRDBNet, SRVGGCompact, RealCUGAN 2x/3x/4x) untuk
  checkpoint yang tidak dikenali Spandrel.
- **`--model_path` + `--arch`** — muat model checkpoint *apapun* langsung,
  dengan opsi hint arsitektur untuk kasus ambigu. Tidak perlu lagi
  mengedit script inference secara manual tiap kali mau coba model baru
  dari OpenModelDB.
- **`--model_url`** — arahkan ke sebuah link, bukan file lokal. Link share
  Google Drive (cara paling umum model OpenModelDB/komunitas didistribusikan)
  otomatis dikonversi jadi direct-download link dan disimpan cache di
  `weights/from_url/` agar bisa dipakai ulang.
- **`.pth` / `.safetensors` / `.onnx` / `.ckpt`** didukung langsung tanpa
  konfigurasi tambahan. Model ONNX berjalan lewat sesi ONNXRuntime terpisah
  (lihat `load_onnx_model()`), karena graph ONNX bukanlah state_dict
  PyTorch dan pantas mendapat jalur kode sendiri yang jujur, bukan dipaksa
  lewat detektor yang sama.
- **Pemulihan OOM otomatis** — kalau satu tile kehabisan VRAM saat diproses,
  sistem otomatis mencoba ulang dengan ukuran sub-tile yang lebih kecil,
  alih-alih membuat output rusak atau seluruh proses crash. Memory juga
  otomatis dibersihkan di antara gambar saat batch processing untuk
  mengurangi fragmentasi.
- **`--log_level` / `--verbose`** — logging terstruktur dengan timestamp,
  bisa diatur dari mode senyap (cuma error) sampai mode penuh detail
  (setiap skor deteksi arsitektur, cache hit, dan percobaan retry).
- **Siap Python 3.11 / 3.12 / 3.13** — `requirements.txt` dan `setup.py`
  sudah diperbarui; `basicsr` bukan lagi dependency wajib untuk inference
  (masih dipakai untuk pipeline training asli, sekarang jadi extra opsional
  lewat `pip install .[train]` — lihat [Training](docs/Training.md)).
- **`test_universal_loader.py`** — script sanity-check: memverifikasi
  environment kamu (torch/spandrel/CUDA), fallback registry, dan bisa
  memuat + menjalankan forward pass dummy pada checkpoint asli sebelum
  kamu percaya penuh untuk dipakai produksi.
- **Toggle fp16/fp32**, tiling, pre-padding, dan opsi inferensi asli
  Real-ESRGAN lainnya tetap dipertahankan seperti semula.

Lihat **[VERSION.md](VERSION.md)** untuk riwayat versi lengkap.

---

## 🔧 Dependensi dan Instalasi

- Python 3.11, 3.12, atau 3.13
- [PyTorch >= 2.2](https://pytorch.org/) (disarankan versi CUDA untuk inferensi GPU)

### Instalasi

```bash
git clone https://github.com/Hanzet22/HG-ESR-NET.git
cd HG-ESR-NET
pip install -r requirements.txt
```

Extras untuk training (hanya diperlukan kalau mau fine-tuning, bukan untuk inferensi):

```bash
pip install .[train]   # menarik basicsr, facexlib, gfpgan
```

### Sanity check sebelum pemakaian pertama

```bash
python test_universal_loader.py
# setelah lolos, coba arahkan ke checkpoint asli:
python test_universal_loader.py --model_path weights/model_kamu.pth
```

---

## ⚡ Quick Inference

### Menggunakan model apapun (OpenModelDB / HuggingFace / hasil training sendiri)

```bash
python inference_realesrgan.py -i inputs/ --model_path weights/model_kamu.pth -o results/
```

Arsitektur dan scale terdeteksi otomatis. Kalau deteksinya ambigu, paksa manual:

```bash
python inference_realesrgan.py -i inputs/ --model_path weights/model.pth --arch rrdbnet --scale 4
```

Lihat daftar arsitektur yang dikenali fallback registry:

```bash
python inference_realesrgan.py --list_archs
```

### Menggunakan model dari link (auto-download)

```bash
python inference_realesrgan.py -i inputs/ --model_url "https://drive.google.com/file/d/XXXXX/view?usp=sharing" -o results/
```

Link share Google Drive otomatis dikonversi jadi direct-download link. File
yang terdownload disimpan cache di `weights/from_url/`, jadi kalau
dijalankan ulang dengan `--model_url` yang sama, langsung pakai cache tanpa
download ulang.

### Menggunakan checkpoint resmi Real-ESRGAN (auto-download)

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```

```console
Cara pakai: python inference_realesrgan.py -n NAMA_MODEL -i infile -o outfile [opsi]...
       atau: python inference_realesrgan.py --model_path PATH -i infile -o outfile [opsi]...
       atau: python inference_realesrgan.py --model_url URL -i infile -o outfile [opsi]...

  -h                   tampilkan bantuan ini
  -i --input           Gambar atau folder input. Default: inputs
  -o --output          Folder output. Default: results
  -n --model_name      Nama model resmi (auto-download). Diabaikan kalau --model_path/--model_url diset.
  --model_path         Path ke checkpoint lokal .pth/.ckpt/.pt/.safetensors apapun.
  --model_url           Download model dari link (link share Google Drive otomatis dikonversi). Prioritas di atas --model_path/-n.
  --arch               Hint arsitektur opsional untuk fallback archs2/.
  --scale              Scale upsampling. Biasanya terdeteksi otomatis dari checkpoint.
  --list_archs         Tampilkan nama arsitektur fallback archs2/ lalu keluar.
  -s, --outscale       Scale upsampling akhir gambar. Default: 4
  --suffix             Suffix nama file gambar hasil. Default: out
  -t, --tile           Ukuran tile, 0 berarti tanpa tiling. Default: 0
  --face_enhance       Pakai GFPGAN untuk enhance wajah. Default: False
  --fp32               Pakai presisi fp32 saat inferensi. Default: fp16 (half precision).
  --ext                Ekstensi gambar. Opsi: auto | jpg | png. Default: auto
  --log_level          Level detail log: debug | info | warning | error. Default: info.
  --verbose            Shortcut untuk --log_level debug.
```

Hasil disimpan ke folder `results/`. Kalau satu tile kehabisan VRAM saat
diproses, sistem otomatis mencoba ulang dengan ukuran lebih kecil, bukan
menggagalkan seluruh gambar — detail lengkap lihat [VERSION.md](VERSION.md).

---

## 🏰 Arsitektur yang Didukung

Ditangani otomatis lewat Spandrel (tanpa konfigurasi): ESRGAN/RRDBNet,
SRVGGCompact, SPAN, OmniSR, SAFMN, HAT, DAT, SwinIR, Swin2SR, SRFormer, RGT,
DRCT, ATD, RealCUGAN (2x/2x_fast/3x/4x), PLKSR/RealPLKSR, RestoreFormer,
RetinexFormer, SCUNet, GRL, DITN, SeemoRe, MoSR, MoESR, DCTLSA, dan
lainnya — lihat [daftar arsitektur Spandrel](https://github.com/chaiNNer-org/spandrel#supported-architectures)
untuk daftar terkini yang lengkap.

Fallback tulisan tangan (`archs2/`, dipakai hanya kalau Spandrel tidak
mengenali checkpoint): RRDBNet, SRVGGCompact, RealCUGAN 2x/3x/4x.

Arsitektur dengan lisensi restriktif/non-komersial tersedia lewat package
terpisah `spandrel_extra_arches` — baca catatan lisensi di
[ATTRIBUTIONS.md](ATTRIBUTIONS.md) sebelum memakainya di luar keperluan
pribadi atau non-komersial.

---

## 📧 Kontak

Pertanyaan atau issue soal **HG-ESR-NET**: `farhanzet4@gmail.com` atau
`hypergarudatkj@gmail.com`.

Untuk pertanyaan soal proyek asli Real-ESRGAN yang jadi basis fork ini,
lihat bagian kredit di bawah.

---

## 🙏 Berdasarkan / Kredit

HG-ESR-NET adalah fork dari **Real-ESRGAN**, dengan jalur
inference/model-loading yang dibangun ulang di sekitar detektor arsitektur
universal. Pipeline training asli, paper, dan proyek upstream-nya
sepenuhnya dikreditkan di bawah ini. Atribusi lengkap untuk setiap
arsitektur dan library yang dipakai ada di [ATTRIBUTIONS.md](ATTRIBUTIONS.md)
— baca dulu sebelum mendistribusikan ulang proyek ini.

### Real-ESRGAN (proyek asli)

> Wang, Xintao, Liangbin Xie, Chao Dong, and Ying Shan. "Real-ESRGAN:
> Training real-world blind super-resolution with pure synthetic data."
> ICCVW 2021.

[[Paper](https://arxiv.org/abs/2107.10833)] &emsp; [[Repo Asli](https://github.com/xinntao/Real-ESRGAN)]

[Xintao Wang](https://xinntao.github.io/), Liangbin Xie,
[Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ),
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) —
Tencent ARC Lab; Shenzhen Institutes of Advanced Technology, Chinese Academy
of Sciences

```bibtex
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```

Kontak proyek asli: `xintao.wang@outlook.com` atau `xintaowang@tencent.com`.

### Dependency inti yang jadi dasar proyek ini

- **[BasicSR](https://github.com/XPixelGroup/BasicSR)** (XPixelGroup /
  Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy, Chao
  Dong) — implementasi arsitektur referensi yang jadi dasar fallback
  `archs2/`.
- **[Spandrel](https://github.com/chaiNNer-org/spandrel)** (tim chaiNNer) —
  engine deteksi dan pemuatan arsitektur otomatis utama yang menggerakkan
  dukungan model universal di fork ini.
- **[OpenModelDB](https://openmodeldb.info)** — utility opsional untuk
  browsing/download/convert file checkpoint model.

Lihat **[ATTRIBUTIONS.md](ATTRIBUTIONS.md)** untuk daftar lengkap
arsitektur, author, paper, dan ketentuan lisensi.

### Proyek terkait yang direkomendasikan

- [GFPGAN](https://github.com/TencentARC/GFPGAN) — restorasi wajah praktis
- [BasicSR](https://github.com/XPixelGroup/BasicSR) — toolbox restorasi gambar/video open-source
- [Spandrel](https://github.com/chaiNNer-org/spandrel) — pemuatan model SR/restorasi PyTorch universal
- [OpenModelDB](https://openmodeldb.info) — database model komunitas
