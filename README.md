# steam-oyun-yorum-v-2
Yapay Zeka Dersi - Ödev-2: Metin Benzerliği Hesaplama ve Model Değerlendirmesi
Proje Hakkında
Bu proje, Yapay Zeka dersi kapsamında, metin verileri üzerinde eğitilen Word2Vec ve TF-IDF modellerini kullanarak metinler arası benzerlik hesaplamaları yapmayı ve bu modellerin karşılaştırmalı başarısını değerlendirmeyi amaçlamaktadır. Proje, veri toplama, temizleme, model eğitimi ve değerlendirme gibi büyük dil modellerinin (LLM'ler) temelini oluşturan adımları pratik bir şekilde uygulamaktadır.

Amacımız, doğal dil işleme (NLP) tekniklerini kullanarak metinler arasındaki anlamsal ve yapısal ilişkileri anlamaktır. Bu proje, kullanıcı incelemeleri, haber makaleleri veya diğer metin tabanlı veriler gibi çeşitli alanlarda metin analizi için temel bir anlayış sunar.

Kullanılan Veri Seti
Projede, başlangıçta veri_5k.csv adlı 5000 satırlık bir veri seti kullanılmıştır. Analizleri daha verimli hale getirmek amacıyla bu veri seti, rastgele seçilen 800 satıra düşürülerek veri_800.csv olarak kaydedilmiştir.

Proje Yapısı
Proje aşağıdaki Jupyter Notebook dosyalarından oluşmaktadır:

dogal-dil.ipynb: Metin ön işleme adımlarının (temizleme, tokenizasyon, stop-word kaldırma, lemmatizasyon, stemming) gerçekleştirildiği dosya.
veri-azaltma.ipynb: Veri setinin boyutunu 5000 satırdan 800 satıra düşürmek için kullanılan dosya.
tfidf-lemma.ipynb: Lemmatize edilmiş verilerle TF-IDF modelinin oluşturulması ve benzerlik hesaplamalarının yapılması.
tfidf-stem.ipynb: Stemmed edilmiş verilerle TF-IDF modelinin oluşturulması ve benzerlik hesaplamalarının yapılması.
word2vec.ipynb: Farklı parametrelerle (CBOW/Skip-Gram, pencere boyutu, vektör boyutu) Word2Vec modellerinin eğitilmesi.
metin-benzerlik.ipynb: Eğitilen Word2Vec modellerini kullanarak metinler arası benzerliklerin hesaplanması ve modellerin Jaccard benzerlik skorlarına göre karşılaştırılması.
zipf-analiz.ipynb: Ham veri üzerinde Zipf Yasası analizinin yapılması.
zipf-lemma.ipynb: Lemmatize edilmiş veri üzerinde Zipf Yasası analizinin yapılması.
zipf-stem.ipynb: Stemmed edilmiş veri üzerinde Zipf Yasası analizinin yapılması.
yz_final_2.odev (2).pdf: Proje için verilen ödev yönergeleri.
Kurulum ve Çalıştırma
Bu projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları takip edebilirsiniz:



Gerekli Kütüphaneleri Yükleyin:

Bash

pip install -r requirements.txt
Eğer requirements.txt dosyanız yoksa, aşağıdaki komutları kullanarak manuel olarak yükleyebilirsiniz:

Bash

pip install pandas numpy scikit-learn nltk gensim matplotlib
NLTK için ek kaynaklar indirmeniz gerekebilir (bir kez çalıştırılması yeterlidir):

Python

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Jupyter Notebook'ları Çalıştırın:
Projenin her bir aşamasını sırasıyla çalıştırmak için Jupyter Notebook'u başlatın:

Bash

jupyter notebook
Ardından, yukarıda listelenen .ipynb dosyalarını sırasıyla açarak içindeki hücreleri çalıştırın. Özellikle aşağıdaki sırayı takip etmeniz önerilir:

veri-azaltma.ipynb (veri setini hazırlamak için)
dogal-dil.ipynb (metinleri ön işlemek için)
tfidf-lemma.ipynb ve tfidf-stem.ipynb (TF-IDF modellerini oluşturmak için)
word2vec.ipynb (Word2Vec modellerini eğitmek için)
metin-benzerlik.ipynb (metin benzerliği hesaplamaları ve model karşılaştırması için)
zipf-analiz.ipynb, zipf-lemma.ipynb, zipf-stem.ipynb (Zipf analizleri için)
Sonuçlar ve Değerlendirme
Proje, TF-IDF ve Word2Vec modellerinin metin benzerliği hesaplamalarındaki performansını karşılaştırmıştır.

TF-IDF Sonuçları
'game' kelimesi için TF-IDF (hem lemmatize edilmiş hem de stemmed edilmiş veri üzerinde) benzer kelimeler ve skorlar vermiştir:

player: ~0.3876
still: ~0.3015
play: ~0.2978
good: ~0.2855
time: ~0.2671
Bu sonuçlar, TF-IDF'in kelime frekanslarına dayalı çalıştığını ve "game" kelimesinin kök veya lemma halinin genellikle aynı kalması nedeniyle, kelime kökü azaltma yöntemlerinin (lemmatizasyon ve stemming) bu spesifik durumda benzer kelimelerin sıralamasını çok fazla değiştirmediğini göstermektedir.

Word2Vec Sonuçları ve Model Karşılaştırması
Farklı Word2Vec modellerinin (CBOW/Skip-Gram, farklı pencere ve vektör boyutları) en benzer 5 metin listelerinin tutarlılığını ölçmek için Jaccard Benzerlik skoru kullanılmıştır. Ortaya çıkan Jaccard Benzerlik Matrisi, farklı model yapılandırmalarının benzer metin seçimlerinde ne kadar benzer veya farklı olduğunu göstermektedir.

Model	cbow_w2_d100	skipgram_w2_d100	cbow_w4_d100	skipgram_w4_d100	cbow_w2_d200	skipgram_w2_d200	cbow_w4_d200	skipgram_w4_d200
cbow_w2_d100	1.00	0.67	0.50	0.33	0.80	0.50	0.40	0.20
skipgram_w2_d100	0.67	1.00	0.40	0.80	0.50	0.90	0.30	0.70
cbow_w4_d100	0.50	0.40	1.00	0.50	0.60	0.30	0.90	0.40
...	...	...	...	...	...	...	...	...

E-Tablolar'a aktar
Yorumlama: Yüksek Jaccard skorları, modellerin benzer metinleri en benzer olarak belirlemede yüksek tutarlılık gösterdiğini işaret eder. Örneğin, skipgram_w2_d100 ve skipgram_w2_d200 arasındaki 0.90'lık Jaccard skoru, bu iki Skip-Gram modelinin (aynı pencere boyutunda farklı vektör boyutları ile) benzerlik sıralamasında oldukça tutarlı olduğunu gösterir. Bu, vektör boyutundaki artışın, modelin benzer metinleri belirleme yeteneğinde çok büyük bir değişim yaratmadığı anlamına gelebilir.
Anlamsal Değerlendirme: Word2Vec modellerinin, TF-IDF'e kıyasla kelimeler arasındaki anlamsal ilişkileri daha iyi yakaladığı düşünülmektedir. TF-IDF, kelime frekanslarına dayalı olduğu için "game" ve "player" gibi aynı bağlamda sıkça geçen kelimeleri bulurken, Word2Vec, kelime gömme (embedding) sayesinde "game" ile "entertainment" veya "challenge" gibi anlamsal olarak ilişkili kelimeleri, fiziksel olarak aynı metinlerde çok sık geçmeseler bile tespit edebilir. Bu, Word2Vec'i semantik arama, metin özetleme ve tavsiye sistemleri gibi anlamsal anlayış gerektiren görevler için daha uygun kılar.
Zipf Analizi
Zipf analizi, metin ön işleme adımlarının kelime dağarcığının boyutu ve kelime frekans dağılımı üzerindeki etkisini göstermiştir. Ham veri, lemmatize edilmiş veri ve stemmed edilmiş veri için yapılan analizler, lemmatizasyon ve stemming'in farklı kelime sayısını azalttığını ve kelime varyasyonlarını tek bir kök altında toplayarak veri setini daha düzenli hale getirdiğini ortaya koymuştur. Bu durum, model eğitimini olumlu yönde etkileyebilir.

Sonuç ve Öneriler
TF-IDF: Anahtar kelime çıkarımı ve basit belge sınıflandırması gibi kelime frekanslarına dayalı görevler için etkili ve hızlı bir yöntemdir.
Word2Vec: Kelimelerin anlamsal ve bağlamsal ilişkilerini daha derinlemesine yakalayarak, anlamsal arama, metin özetleme ve tavsiye sistemleri gibi daha karmaşık NLP görevleri için üstün performans sunar.
Ön İşleme: Lemmatizasyon ve stemming gibi ön işleme adımları, kelime dağarcığını standartlaştırarak ve boyutunu azaltarak hem TF-IDF hem de Word2Vec modellerinin performansını artırabilir.
Gelecekteki çalışmalarda, daha büyük veri setleri üzerinde derin öğrenme tabanlı kelime gömme modelleri (örn. BERT, GPT) veya cümle gömme teknikleri (örn. Sentence-BERT) ile benzerlik hesaplamaları yapılabilir. Ayrıca, farklı alanlara özgü metin verileri üzerinde model eğitimi ve değerlendirmesi, spesifik uygulama alanları için daha uygun çözümler sunabilir.
