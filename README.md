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

## Windows: hazir surum (Python gerekmez)

`main`'e her push'ta GitHub, Windows icin `VisionPlay.exe`'yi kendisi derler
ve once oyunun Windows'ta calistigini test eder.

1. Depo sayfasinda **Actions** sekmesine gecin, soldan **Windows**'u secin.
2. En ustteki yesil tikli calismaya girin; sayfanin altindaki **Artifacts**
   bolumunden **VisionPlay-Windows**'u indirin (GitHub'a giris yapmis olmak
   gerekir).
3. Zip'i bir klasore cikarin ve `VisionPlay.exe`'yi calistirin. Yanindaki
   `_internal` klasoru yerinde kalmali.

Imzasiz programlarda Windows "bilgisayarinizi korudu" uyarisi verir:
**Ek bilgi > Yine de calistir**. Yeni surum icin ayni yerden tekrar indirin.

## Kaynaktan kurulum

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

Baslangic ekranindaki **KAMERA** dugmesi takili kameralar arasinda gecer;
secilen kamera bir sonraki acilista da kullanilir.

### Telefonu kamera olarak kullanma

- **Windows 11 + Android:** Ayarlar > Bluetooth ve cihazlar > Mobil cihazlar'dan
  telefonu eslestirip "bagli kamera olarak kullan"i acin.
- **Diger telefonlar (iPhone dahil):** Iriun Webcam, DroidCam ya da Camo gibi bir
  uygulamayi hem telefona hem bilgisayara kurun.

Telefon bilgisayara yeni bir kamera olarak eklenir; oyunda KAMERA dugmesiyle ona
gecin. Telefonu yatay tutun, gecikme daha az oldugu icin mumkunse USB ile baglayin.

Tam ekran icin `f`, cikmak icin `q` tusuna basin. Pencere yeniden
boyutlandirilabilir ve goruntu orani korunur.

## Uygulama olarak paketleme

Sanal ortam etkinken (basta `(venv)` yazarken):

```bash
python -m pip install pyinstaller==6.22.3
python -m PyInstaller VisionPlay.spec --noconfirm --clean
```

`python -m` bicimi, derlemeyi oyunun paketlerinin kurulu oldugu Python'la
yaptirir. Duz `pyinstaller` komutu bilgisayardaki baska bir Python'a denk
gelirse `.exe` "No module named 'mediapipe'" hatasiyla acilmaz.

macOS'ta sonuc `dist/VisionPlay.app`, Windows'ta `dist/VisionPlay/VisionPlay.exe`.
PyInstaller bir sistemden digeri icin derleyemez; Windows surumunu
`.github/workflows/windows.yml` GitHub'in Windows makinesinde uretir.

## Proje yapisi

- `core/` — kamera, el takibi, geometri, piksel yazi tipi, ortak arayuz parcalari
- `modes/` — her oyun modu ve menu ekranlari
- `assets/` — Aseprite ile cizilmis piksel gorseller ve yazi tipleri
- `main.py` — ekranlar arasi gecisi yoneten ana dongu

## Gereksinimler

Python 3.11+, bir web kamerasi ve iyi isik.

macOS uzerinde gelistirildi; Windows'ta kaynaktan calisir. Kamera acilisi
Windows'ta DirectShow arka ucu uzerinden yapilir, ayrica bir sey gerekmez.
