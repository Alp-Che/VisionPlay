# VisionPlay

Kamera ve el hareketleriyle oynanan mini oyunlar. Klavyeye veya fareye
dokunmadan, 2-3 metre uzaktan oynanacak sekilde tasarlandi.

Butun arayuz el ile kullaniliyor: bir butonun uzerine parmak ucunuzu getirip
**yumrugunuzu 1 saniye kapali tutunca** tiklaniyor.

## Modlar

| Mod | Nasil oynanir |
|---|---|
| **OKCULUK** | Bir el yayi tutar, diger el yumruk yapinca ok olusur; oku yaya yaklastirip geri cekin ve birakin. Hedef asagi yukari hareket eder. |
| **CIZIM** | Bir elin isaret parmagi kalemdir, diger el yumruk yapinca cizer. YON DEGISTIR iki elin gorevini degistirir. |
| **FRUIT NINJA** | Iki isaret parmaginizla meyveleri kesin, bombalara dokunmayin. Ust uste kestikce kombo buyur; saat kesince sure kazanirsiniz. |
| **TOP SEKTIRME** | Topu elinizle havada tutun, yere dusurmeyin. |
| **KOSTEBEK** | Alttan cikan kostebekleri elinizle geri sokun, kotu olanlara vurmayin. |
| **PONG** | Iki kisilik. Herkes kendi tarafindaki raketi eliyle surer. |
| **DANS** | Duvar geliyor; iki elinizi deliklerine yerlestirip gecin. Delikler bir sagda bir solda, gitgide hizlanir. |

Oyun bitince rekorunuz da gosterilir (TOP SEKTIRME icin en uzun seri, sag ustte). Rekorlar bilgisayarda saklanir (macOS: `~/Library/Application Support/VisionPlay`, Windows: `%APPDATA%\VisionPlay`).

## Kurulum

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup_models.py
```

**Windows** (PowerShell)

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup_models.py
```

PowerShell betik calistirmayi engellerse, `venv\Scripts\activate.bat` dosyasini
normal komut isteminde (cmd) calistirin.

`setup_models.py`, MediaPipe'in el takibi modelini `models/` klasorune indirir.
Model dosyasi depoya dahil degildir, bu adim atlanirsa oyun acilmaz.

## Calistirma

```bash
python main.py
```

Sanal ortamin etkin oldugundan emin olun (komut satirinin basinda `(venv)`
yaziyor olmali).

Ilk calistirmada isletim sistemi kamera izni ister. Windows'ta izin
**Ayarlar > Gizlilik ve guvenlik > Kamera** altindadir; "Masaustu uygulamalarinin
kameraniza erismesine izin verin" acik olmalidir.

Tam ekran icin `f`, cikmak icin `q` tusuna basin. Pencere yeniden
boyutlandirilabilir ve goruntu orani korunur.

## macOS uygulamasi olarak paketleme

```bash
pip install pyinstaller
pyinstaller VisionPlay.spec --noconfirm
```

Sonuc: `dist/VisionPlay.app`

Bu tarif yalnizca macOS icindir: `.icns` ikon ve `BUNDLE` adimi macOS'a ozgudur.
Windows'ta oyun kaynaktan `python main.py` ile calistirilir.

## Proje yapisi

- `core/` — kamera, el takibi, geometri, piksel yazi tipi, ortak arayuz parcalari
- `modes/` — her oyun modu ve menu ekranlari
- `assets/` — Aseprite ile cizilmis piksel gorseller ve yazi tipleri
- `main.py` — ekranlar arasi gecisi yoneten ana dongu

## Gereksinimler

Python 3.11+, bir web kamerasi ve iyi isik.

macOS uzerinde gelistirildi; Windows'ta kaynaktan calisir. Kamera acilisi
Windows'ta DirectShow arka ucu uzerinden yapilir, ayrica bir sey gerekmez.
