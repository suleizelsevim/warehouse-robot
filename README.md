## Warehouse Robot – Palet Toplama
Bu repo, 6×6 bir depo ortamında çalışan Q-Learning tabanlı otonom bir depo robotu uygulamasını içeriyor.
## Nasıl Çalıştırılır
Gerekli paketleri kurun
```bash
pip install numpy matplotlib pillow
```
Ardından kodu çalıştırın
```bash
python warehouse-robot.py
```
## Ortam tasarımı
Ortam 6x6 bir grid:

Robot başlangıcı → (0,0)
Çıkış noktası → (5,5)

3 palet → her episode’da rastgele

Robot dört yöne hareket edebilir

Palet üstüne gelindiğinde otomatik olarak toplanır

## Aksiyon Uzayı
| Aksiyon | Açıklama |
| ------- | -------- |
| 0       | Yukarı     |
| 1       | Aşağı   |
| 2       | Sola   |
| 3       | Yukarı   |

Pickup aksiyonu yoktur, paletin üstüne gelmek yeterlidir.

## Ödül Sistemi

| Parametre | Değer |    Açıklama |
| ------- | -------- |   ------    |
| STEP_PENALTY       | -0.1     |  Her adım için küçük ceza |
| PICKUP_REWARD     | +20   | Palet toplama ödülü  |
| DELIVERY_REWARD       |+50   | Teslimat ödülü  |

## Q-Learning
Depo ortamında durum uzayı çok geniş olduğu için Q tablosu, numpy matrisi yerine dictionary tabanlı tutulmaktadır.

Bir state şu durumlardan oluşur:
```bash
(robot_r, robot_c, pallet_locations, collected_mask)
```
Bu sayede yalnızca gerçekten ziyaret edilen durumlar Q-tablosunda yer alır.

| Parametre | Değer      | Açıklama                  |
| --------- | ---------- | ------------------------- |
| alpha     | 0.7        | Öğrenme oranı             |
| gamma     | 0.93      | Geleceğe verilen önem     |
| epsilon   | 1.0 → 0.05 | Lineer azalır             |
| episodes  | 170000     | Eğitim sayısı       |
| max_steps | 250        | Her bölümde maksimum adım |

## Test Sonuçları

Eğitilen ajan 100 episode boyunca test edildi

=== EVALUATION RESULTS ===
Episodes       : 100
Success count  : 100
Success rate   : 100.0%
Avg reward     : 108.59
>Ajan %100 başarı oranına ulaşmıştır.

![](https://github.com/suleizelsevim/warehouse-robot/blob/master/warehouse.gif)



