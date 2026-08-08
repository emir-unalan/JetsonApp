# Gereksinimler

JetsonApp, NVIDIA Jetson platformları için geliştirilmiştir ve CUDA destekli OpenCV ile TensorRT kullanmaktadır. Bu nedenle sistemde aşağıdaki bileşenlerin kurulu olması gerekmektedir.

## Donanım

Proje aşağıdaki donanım üzerinde geliştirilmiş ve test edilmiştir.

| Donanım | Durum |
|---------|-------|
| Jetson Orin Nano Super Developer Kit | ✅ Test edildi |

Farklı Jetson kartlarında da çalışması beklenmektedir. Ancak TensorRT Engine dosyası hedef cihaz üzerinde yeniden oluşturulmalıdır.

---

## Yazılım

| Yazılım | Sürüm |
|----------|-------|
| Ubuntu | JetPack ile gelen Ubuntu |
| JetPack | **6.2.2** |
| CUDA | **12.6** |
| TensorRT | JetPack ile gelen sürüm |
| OpenCV | **4.13.0 (CUDA Build)** |
| CMake | 3.16 veya üzeri |
| GCC | JetPack varsayılan GCC |
| CLI11 | CMake `FetchContent` ile otomatik indirilir (ayrı kurulum gerekmez) |

---

# OpenCV 4.13 (CUDA) Kurulumu

JetsonApp, sistemde bulunan standart OpenCV paketi ile çalışacak şekilde tasarlanmamıştır.

Projede;

- CUDA Image Processing
- CUDA Resize
- CUDA Color Conversion
- CUDA Memory Operations
- GStreamer
- TensorRT ile GPU tabanlı veri akışı

kullanıldığı için **OpenCV 4.13.0'ın CUDA desteği ile kaynak koddan derlenmesi gerekmektedir.**

---

## Gerekli Paketlerin Kurulumu

```bash
sudo apt update

sudo apt install -y \
    build-essential cmake git pkg-config \
    libgtk-3-dev libcanberra-gtk3-module \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libtbb-dev libatlas-base-dev gfortran \
    python3-dev python3-numpy \
    libjpeg-dev libpng-dev libtiff-dev \
    libopenblas-dev liblapack-dev
```

---

## OpenCV Kaynak Kodunun İndirilmesi

```bash
mkdir -p ~/src

cd ~/src

git clone --branch 4.13.0 --depth 1 https://github.com/opencv/opencv.git

git clone --branch 4.13.0 --depth 1 https://github.com/opencv/opencv_contrib.git
```

---

## Build Dizininin Oluşturulması

```bash
mkdir -p ~/src/opencv/build

cd ~/src/opencv/build
```

---

## CMake Konfigürasyonu

```bash
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=~/src/opencv_contrib/modules \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D WITH_CUBLAS=ON \
-D CUDA_ARCH_BIN=8.7 \
-D WITH_GSTREAMER=ON \
-D WITH_LIBV4L=ON \
-D BUILD_opencv_python3=ON \
-D BUILD_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D OPENCV_GENERATE_PKGCONFIG=ON \
..
```

---

## Derleme

Jetson Orin Nano Super üzerinde aşağıdaki komut önerilmektedir.

```bash
make -j4
```

Derleme tamamlandıktan sonra:

```bash
sudo make install

sudo ldconfig
```

---

# ⚠️ Önemli Not: Bellek (RAM + Swap)

CUDA destekli OpenCV'nin derlenmesi sırasında derleyici yüksek miktarda bellek kullanmaktadır.

Derlemeye başlamadan önce sisteminizde yeterli toplam bellek bulunduğundan emin olun.

Toplam RAM ve Swap miktarını görmek için:

```bash
free -h
```

Swap durumunu görmek için:

```bash
swapon --show
```

JetPack 6.2.2'nin varsayılan kurulumunda **swap etkin olarak gelmektedir**.

Eğer sistem üzerinde swap yapılandırmasını değiştirmediyseniz (örneğin `nvzramconfig` ayarları ile oynamadıysanız), ek bir swap oluşturmanıza gerek yoktur. Varsayılan yapılandırma ile `make -j4` komutu güvenli şekilde tamamlanmaktadır.

Ancak;

- swap devre dışı bırakıldıysa,
- toplam kullanılabilir bellek yetersizse,
- veya çok yüksek paralellikte (`make -j8`, `make -j12` gibi) derleme yapılırsa,

derleme sırasında aşağıdaki hatalardan biri alınabilir.

```text
cc1plus: fatal error: Killed signal terminated program cc1plus
```

veya

```text
Out Of Memory
```

Bu durumda derlemeyi daha düşük iş parçacığı sayısı ile (`make -j2`) tekrar başlatmanız önerilir.

---

# OpenCV Kurulumunun Doğrulanması

Kurulumun başarılı olduğunu doğrulamak için aşağıdaki komutu çalıştırabilirsiniz.

```bash
pkg-config --modversion opencv4
```

Beklenen çıktı:

```text
4.13.0
```

CUDA desteğini doğrulamak için ise:

```bash
opencv_version
```

veya

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())"
```

çıktısında aşağıdaki satırların bulunması gerekir.

```text
CUDA: YES
```

```text
cuDNN: YES
```

```text
GStreamer: YES
```

```text
NVIDIA CUDA: YES
```

---

# JetsonApp'ın Derlenmesi

OpenCV kurulumu tamamlandıktan sonra proje kendi CMake yapılandırmasıyla derlenir. CLI11 kütüphanesi `FetchContent` ile otomatik indirildiği için önceden kurulum yapmanız gerekmez (internet bağlantısı gerekir).

```bash
mkdir -p build && cd build

cmake ..

make -j4
```

Derleme sonunda çalıştırılabilir dosya `build/jetsonApp` altında oluşur.

---

# Kullanım

JetsonApp, komut satırı argümanlarını **CLI11** kütüphanesi ile ayrıştırır. Belirtilmeyen argümanlar için varsayılan değerler, çalıştırılabilir dosyanın bulunduğu dizine göre otomatik belirlenir.

```bash
./jetsonApp [seçenekler]
```

### Argümanlar

| Flag | Açıklama | Varsayılan |
|---|---|---|
| `-e, --engine <path>` | TensorRT engine dosyasının yolu | `<exe_dizini>/models/model.engine` |
| `-o, --onnx <path>` | ONNX model dosyasının yolu | `<exe_dizini>/models/model.onnx` |
| `-s, --source <path>` | Video kaynağı dosya yolu | `<exe_dizini>/video.mkv` |
| `-c, --cam` | Video dosyası yerine kamerayı kaynak olarak kullanır (`nvarguscamerasrc` tabanlı GStreamer pipeline'ı) | kapalı |
| `--save` | İşlenmiş kareleri `output/` klasörüne JPEG olarak kaydeder | açık |
| `--show` | Kareleri bir pencerede canlı gösterir (SSH/headless CLI oturumunda kullanmayın) | kapalı |
| `-v, --verbose` | Her 60 karede bir preprocess/infer/postprocess sürelerini ve FPS'i konsola yazdırır | kapalı |

> Engine dosyası bulunamazsa, program ONNX dosyasından yeni bir engine oluşturur. Bu işlem yalnızca ilk çalıştırmada veya model değiştiğinde birkaç dakika sürer.

### Örnekler

Varsayılan ayarlarla (repodaki `video.mkv` üzerinden, kareleri kaydederek) çalıştırma:

```bash
./jetsonApp
```

Özel bir video dosyası ve model ile, detaylı loglama açık:

```bash
./jetsonApp -s /path/to/video.mp4 -e models/custom.engine -o models/custom.onnx -v
```

Jetson kamerasını canlı kaynak olarak kullanıp ekranda gösterme (yalnızca masaüstü/monitör bağlıyken):

```bash
./jetsonApp --cam --show
```

Kare kaydetmeden, sadece performans ölçümü için:

```bash
./jetsonApp --cam -v
```

---

# Kaynak Kod Yapısı

JetsonApp modüler bir mimari ile geliştirilmiştir. Her dosya belirli bir sorumluluğa sahiptir.

| Dosya | Açıklama |
|--------|----------|
| `main.cpp` | Uygulamanın giriş noktasıdır. CLI11 ile argüman ayrıştırma, model yükleme, video/kamera okuma ve inference döngüsünü yönetir. |
| `inference.cpp` | TensorRT Engine oluşturma, yükleme ve inference işlemlerini gerçekleştirir. |
| `inference.hpp` | `Engine` sınıfının tanımları bulunur. |
| `preprocess.cu` | CUDA tabanlı preprocessing kernel'larını içerir (resize + BGR→RGB + normalize + HWC→CHW, tek geçişte). |
| `preprocess.cuh` | Preprocessing fonksiyonlarının tanımları. |
| `postprocess.cu` | CUDA tabanlı decode ve postprocessing işlemleri. |
| `postprocess.cuh` | Postprocessing fonksiyonlarının tanımları. |

Bu yapı sayesinde proje kolayca genişletilebilir ve farklı modeller için uyarlanabilir.

---

# Bellek Yönetimi

JetsonApp performans kaybını azaltmak amacıyla gereksiz bellek kopyalamalarını önleyecek şekilde tasarlanmıştır.

Temel prensip:

```text
CPU

↓

GPU Upload

↓

GPU Memory

↓

CUDA Kernels

↓

TensorRT

↓

CUDA Postprocess

↓

Minimum CPU Transfer
```

Bu yaklaşım sayesinde;

- daha düşük gecikme,
- daha düşük CPU kullanımı,
- daha yüksek FPS

elde edilmektedir.

---

# Desteklenen Model Formatları

JetsonApp TensorRT tarafından desteklenen ONNX modelleri ile çalışmaktadır.

Örneğin;

- YOLOv8
- YOLOv9
- YOLOv10
- YOLO11

ve benzeri ONNX formatına dönüştürülebilen modeller kullanılabilir.

Model değiştirildiğinde mevcut `.engine` dosyası silinmeli ve uygulama yeniden çalıştırılmalıdır (veya `--engine`/`--onnx` argümanlarıyla farklı bir dosya çifti belirtilmelidir). Böylece TensorRT yeni modele uygun engine dosyasını yeniden oluşturacaktır.

---

# Performans Önerileri

En yüksek performansı elde etmek için aşağıdaki öneriler dikkate alınabilir.

## Güç Modunu Maksimuma Alın

Jetson cihazlarda maksimum performans modu önerilir.

```bash
sudo nvpmodel -m 0

sudo jetson_clocks
```

Bu komutlar CPU, GPU ve bellek frekanslarının maksimum performans seviyesinde çalışmasını sağlar.

---

## FP16 Kullanın

TensorRT FP16 desteği bulunan modellerde önemli performans artışı sağlar.

JetsonApp desteklenen cihazlarda FP16 Engine oluşturmaktadır.

---

## CUDA Destekli OpenCV Kullanın

Sistem paket yöneticisi ile kurulan standart OpenCV sürümleri CUDA modüllerini içermez.

Bu nedenle README içerisinde anlatıldığı şekilde OpenCV 4.13.0'ın CUDA desteği ile derlenmesi önerilmektedir.

---

## Engine Dosyasını Tekrar Oluşturmayın

Engine oluşturma işlemi yalnızca model değiştiğinde gereklidir.

Her çalıştırmada yeniden engine oluşturmak uygulamanın başlangıç süresini ciddi şekilde artıracaktır.

---

## Gereksiz Kare Kaydını Kapatın

`--save` varsayılan olarak açıktır ve her kareyi diske JPEG olarak yazar. Sadece canlı izleme veya performans testi yapıyorsanız `--save` bayrağını kullanmayarak (kodda varsayılanı değiştirerek) disk I/O yükünü azaltabilirsiniz.

---

# Sık Karşılaşılan Sorunlar

## OpenCV Bulunamıyor

Hata:

```text
Could NOT find OpenCV
```

Çözüm:

- OpenCV'nin `/usr/local` altına kurulduğunu doğrulayın.
- `sudo ldconfig` komutunu çalıştırın.
- Gerekirse `build` klasörünü silerek CMake'i yeniden çalıştırın.

---

## CUDA Bulunamıyor

Hata:

```text
CUDA NOT FOUND
```

Çözüm:

CUDA'nın doğru kurulduğunu kontrol edin. 

```bash
nvcc --version
```
Hata mesajını inceleyerek eklemeniz gereken bir CMAKE configrasyonu var mı kontrol edin.

---

## CLI11 İndirilemiyor

Hata:

```text
Failed to download content of 'CLI11'
```

Çözüm:

- Cihazın internet bağlantısını kontrol edin (CLI11, CMake `FetchContent` ile GitHub üzerinden indirilir).
- Offline ortamda çalışıyorsanız CLI11'i önceden indirip `third_party/CLI11` altına yerleştirip CMakeLists.txt'te `add_subdirectory` ile eklemeyi değerlendirin.

---

## Video/Kamera Açılamıyor

Hata:

```text
Hata: Video açılamadı.
```

Çözüm:

- `-s/--source` ile verilen video dosyasının yolunun doğru olduğunu kontrol edin.
- `-c/--cam` kullanıyorsanız, `nvarguscamerasrc` pipeline'ının cihazınızdaki kamera sensörüyle (çözünürlük/framerate) uyumlu olduğunu doğrulayın.
- GStreamer desteğinin OpenCV derlemesinde etkin olduğunu (`WITH_GSTREAMER=ON`) doğrulayın.

---

## Engine Oluşturulamıyor

Olası nedenler:

- ONNX modeli bozuk olabilir.
- TensorRT tarafından desteklenmeyen operatörler kullanılabilir.
- Yanlış CUDA sürümü ile oluşturulmuş eski `.engine` dosyası kullanılmaya çalışılıyor olabilir.

Çözüm olarak mevcut `.engine` dosyasını silip uygulamayı yeniden çalıştırın.

---

## Düşük FPS

Aşağıdaki maddeleri kontrol edin.

- Güç modu maksimum seviyede mi?
- `jetson_clocks` aktif mi?
- OpenCV CUDA desteği ile derlendi mi?
- FP16 Engine kullanılıyor mu?
- Kamera çözünürlüğü gereğinden yüksek mi?
- `-v/--verbose` ile preprocess/infer/postprocess sürelerini karşılaştırarak darboğazın hangi aşamada olduğunu tespit edin.

---

# Geliştirme Planı (Roadmap)

Planlanan geliştirmeler:

- [ ] INT8 Quantization desteği
- [ ] Çoklu kamera desteği
- [ ] RTSP desteği
- [ ] USB kamera desteğinin genişletilmesi
- [ ] Batch inference desteği
- [ ] CUDA Graph desteği
- [ ] Asenkron inference pipeline
- [ ] Çoklu TensorRT Engine desteği
- [ ] ROS 2 entegrasyonu
- [ ] GStreamer pipeline desteğinin geliştirilmesi
- [ ] Deepstream Desteği ile Object Tracking
- [ ] Farklı NVIDIA ekran kartları üzerinde de programın optimize bir şekilde çalıştırılabilmesi
---

# Katkıda Bulunma

Katkılar memnuniyetle karşılanmaktadır.

Yeni özellik önerileri, hata düzeltmeleri veya performans iyileştirmeleri için Pull Request gönderebilir veya Issue oluşturabilirsiniz.

Lütfen katkı göndermeden önce kodun mevcut yapısını korumaya ve mümkün olduğunca modüler geliştirmeler yapmaya özen gösterin.

---

# Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.

Ayrıntılı bilgi için `LICENSE` dosyasına göz atabilirsiniz.

---

# Teşekkürler

Bu proje aşağıdaki açık kaynak teknolojilerden yararlanmaktadır.

- NVIDIA CUDA
- NVIDIA TensorRT
- OpenCV
- CMake
- ONNX
- CLI11
