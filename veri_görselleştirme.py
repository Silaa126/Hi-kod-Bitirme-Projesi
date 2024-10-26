import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# veri yüklemesi yaptık
df = pd.read_csv('sentimentdataset.csv')

# Veriyi incelemeye başlıyoruz
# Verinin ilk satırlarına bakıyoruz 
print(df.head())

# Son beş satırı okuma
print(df.tail())

# Satır ve sütun kontrolü
print(df.columns)
print(df.axes)
print(df.shape)

# Veri tiplerini inceliyor ve bunların beklenen tipler olup olmadığını anlıyoruz
print(df.info())

# Sayısal sütunlar için temel istatisttikleri alıyoruz(ortalama, medyan,min,max vb.)
print(df.describe())

# Eksik verileri kontrol ediyoruz 
# Eksik değerlerin hangi sütunlarda bulunduğunu ve bu eksik verilerin oranlarını tespit edin. 
# Eksik değer analizi
missing_data = df.isnull().sum() / len(df) * 100  # Eksik değerlerin oranını yüzdesel olarak bulma
print("Eksik değer oranları (%):\n", missing_data)

# Eksik veri doldurma
for column in df.columns:
    if df[column].isnull().sum() > 0:  # Sadece eksik veri içeren sütunları kontrol eder
        if missing_data[column] < 5:  # Eksik veri oranı %5'ten az ise
            # Sayısal veri için ortalama, kategorik veri için mod ile doldurma
            if df[column].dtype == 'float64' or df[column].dtype == 'int64':
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            print(f"{column} sütununda eksik veri oranı %5'ten fazla, bu sütunu elle gözden geçirin.")

# Eksik değer kontrolü
print("Doldurma sonrası eksik veri sayıları:\n", df.isnull().sum())

# Eksik veri içeren satır ve sütunları silme
df.dropna(inplace=True)

# Sütunu, tarih ve saat bilgilerini daha kolay işlemek için üç farklı bileşene (gün, ay, yıl) ayrılmak istedim.

df['Timestamp'] = pd.to_datetime(df['Timestamp']) #Timestamp sütununu datetime (tarih ve saat) formatına dönüştürüyor. Eğer Timestamp sütunu tarih formatında değilse, bu dönüşüm veri tipi üzerinde çeşitli zaman bazlı işlemleri yapmanızı sağlar.
df['Day'] = df['Timestamp'].dt.day #Timestamp'deki gün bilgisini (day) çıkarır ve yeni bir sütun olan Day'e ekler. Böylece her zaman damgasından sadece gün bilgisi alınır.
df['Month'] = df['Timestamp'].dt.month #Timestamp'deki ay bilgisini (month) çıkarır ve Month adında yeni bir sütun oluşturur.
df['Year'] = df['Timestamp'].dt.year #Bu satır, Timestamp'den yıl bilgisini (year) çıkarır ve yeni bir Year sütununa ekler.


# Veri setindeki belirli sütunlarda bulunan metinlerden baştaki ve sondaki gereksiz boşluklar temizlenmek istedim

df['Text'] = df['Text'].str.strip() #Text sütunundaki her bir metin değerinin başındaki ve sonundaki boşluklar temizleniyor.
df['Sentiment'] = df['Sentiment'].str.strip() #Yukarıdaki aynı işlem Sentiment sütunundaki metinler için yapılıyor.
df['User'] = df['User'].str.strip() #User sütunundaki kullanıcı adlarındaki gereksiz boşluklar temizleniyor.
df['Platform'] = df['Platform'].str.strip() # Platform sütunundaki sosyal medya platformu adlarındaki boşluklar temizleniyor.
df['Hashtags'] = df['Hashtags'].str.strip() #Hashtags sütununda bulunan hashtag'ler üzerindeki gereksiz boşluklar kaldırılıyor
df['Country'] = df['Country'].str.strip()   #Country sütunundaki ülke isimlerinden boşluklar temizleniyor.


# Bu kodda, Retweets ve Likes sütunlarındaki verilerin veri tipi değiştirilmekte
df["Retweets"] = df["Retweets"].astype(int)
df["Likes"] = df["Likes"].astype(int)

# DataFrame içindeki belirli sütunlar kaldırıldı
df.drop(columns = ["Unnamed: 0.1","Unnamed: 0"], axis = 1, inplace = True)

# Veri çerçevesindeki kategorik (object) veri türüne sahip sütunların 
# istatistiksel özetini verir. k değişkeni bu özeti tutacaktır
k = df.describe(include="object")
print(k)

#"Sentiment" sütunundaki en yaygın 10 değeri ve bu değerlerin frekanslarını 
# içeren bir Pandas veri çerçevesi olur.
sten = df["Sentiment"].value_counts().head(10).reset_index()
sen = pd.DataFrame(sten)
print(sen)

#Platformlar arasındaki dağılımın yüzdelik dilimlerini
colors = ['#808080', '#E1306C', '#1877F2']  # Twitter, Instagram, Facebook için renkler
df['Platform'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=colors)
plt.title('Percentages of Platforms')
plt.legend()
plt.show()

# Bu grafik, farklı duygu etiketlerinin veri setindeki yüzdesel dağılımını gösteren bir duygu
# analizi pasta grafiği
plt.figure(figsize=(10,8))
s = plt.pie(sen["count"], labels = sen["Sentiment"],autopct='%1.1f%%')
plt.show()

# Veri çerçevesinde bulunan hashtag'lerin retweet sayılarıyla ilgili bazı analizler yaparak en çok 
# retweet edilen 10 hashtag'i bulmak için kullanılır.
k = df.groupby("Hashtags")["Retweets"].max().nlargest(10).sort_values(ascending = False).reset_index()
b = pd.DataFrame(k)
print(b)

# Top 10 Hashtags by Retweet Count grafiği oluşturmak
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33B5', '#33FFF2', '#FF8333', '#FF3333', '#B533FF', '#FF3333']

# Çubuk grafiğini oluşturma
plt.figure(figsize=(10, 8))
sns.barplot(x="Retweets", y="Hashtags", data=k, palette=colors)  # Renk paleti olarak colors kullanıldı
for bars in plt.gca().containers:
    plt.bar_label(bars)  # Çubukların üstüne değerleri ekler

plt.title('Top 10 Hashtags by Retweet Count')  # Başlık
plt.xlabel('Retweet Count')  # X ekseni etiketi
plt.ylabel('Hashtags')  # Y ekseni etiketi
plt.show()

#df veri çerçevesinde bulunan hashtag'lerin "Likes" sütunundaki maksimum değerleri bulmak için kullanılır
# ve en çok beğeni alan 10 hashtag'i belirler.
k = df.groupby("Hashtags")["Likes"].max().nlargest(10).reset_index()
b = pd.DataFrame(k)
print(b)

#Top 10 Hashtags by Likes grafiği
colors = ['#4ECDC4', '#FF6F61', '#6A5B8A', '#F7B7A3', '#E5B956', '#D83A56', '#8BC34A', '#FF8A65', '#C2185B', '#6D4C6C']
# Çubuk grafiğini oluşturma
plt.figure(figsize=(8, 6))
sns.barplot(x="Likes", y="Hashtags", data=k, palette=colors)  # Renk paleti olarak colors kullanıldı

plt.title('Top 10 Hashtags by Likes')  # Başlık
plt.xlabel('Likes')  # X ekseni etiketi
plt.ylabel('Hashtags')  # Y ekseni etiketi
plt.show()


# Yıllara göre likes sayılarının analizi
a = df.groupby('Year')['Likes'].sum().reset_index()
ly = pd.DataFrame(a)
print(ly)

plt.figure(figsize = (8,6))
sns.lineplot(x = "Year", y = "Likes", data = ly, marker = "o")
plt.show()

# Sosyal medya platformlarındaki beğeni sayısının saatlere göre yoğunluğunu gösteren bir histogram 
# çıkarttık
plt.figure(figsize=(8,6))
sns.scatterplot(data = df, x = 'Hour', y = 'Likes',hue = "Platform")
plt.show()

# Sosyal medya platformlarındaki 
twitter_data = df[df['Platform'] == 'Twitter']

# Scatter plot oluşturma
plt.figure(figsize=(8, 6))
sns.scatterplot(data=twitter_data, x='Retweets', y='Year', hue='Platform', palette=['#1DA1F2'])  # Twitter rengi
plt.title('Number of Retweets by Year (Just Twitter)')
plt.xlabel('Retweet Count')
plt.ylabel('Year')
plt.legend(title='Platform')
plt.show()

# Aykırı veri analizi: verinin dağılımını ve aykırı değerleri görmek için görselleştirmeler eklenmiş. 
# Aykırı değerler özellikle Retweets, Likes gibi etkileşim verilerinde önemlidir, çünkü analizde 
# sapmaya yol açabilir. Bu nedenle bu değerler görsele dökülmüştür.
# Likes sütunu için boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x='Likes', data=df)
plt.title("Likes Sütunu Aykırı Değer Analizi")
plt.show()

# Retweets sütunu için boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x='Retweets', data=df)
plt.title("Retweets Sütunu Aykırı Değer Analizi")
plt.show()

# Histogram for Retweets
plt.figure(figsize=(10, 5))
sns.histplot(df['Retweets'], bins=30, kde=True)
plt.title("Retweets Sütunu Histogramı")
plt.show()


# Histogram for Likes
plt.figure(figsize=(10, 5))
sns.histplot(df['Likes'], bins=30, kde=True)
plt.title("Likes Sütunu Histogramı")
plt.show()

# Metin sütununda duygu durumları etiketlenmiş (Pozitif, Negatif, Nötr). 
# Duygu analizi sonuçlarının kelime dağılımlarıyla birlikte değerlendirilmesi 
# verinin daha iyi anlaşılmasını sağlar. Bu nedenle burada bu durumun analizi yapılmıştır.
# Positive Sentiment için Word Cloud
from wordcloud import WordCloud
positive_text = " ".join(df[df['Sentiment'] == 'Positive']['Text'])
wordcloud = WordCloud().generate(positive_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Positive Sentiment Word Cloud")
plt.axis("off")
plt.show()     

# Negative Sentiment için Word Cloud
negative_text = " ".join(df[df['Sentiment'] == 'Negative']['Text'])
wordcloud_negative = WordCloud().generate(negative_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis("off")
plt.title("Negative Sentiment Word Cloud")
plt.show()

# Neutral Sentiment için Word Cloud
neutral_text = " ".join(df[df['Sentiment'] == 'Neutral']['Text'])
wordcloud_neutral = WordCloud().generate(neutral_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis("off")
plt.title("Neutral Sentiment Word Cloud")
plt.show()

# Platform ve Etkileşim Analizi:
# Platform değişkenine göre beğeni, retweet ve duygu durumlarının nasıl dağıldığını incelemek, 
# hangi platformda daha fazla etkileşim alındığını veya kullanıcıların duygu durumlarını belirleyebilir.
#Bu nedenle platform bazında groupby() ile etkileşimleri analiz edebilir ve görselleştirilir.
platform_likes = df.groupby('Platform')['Likes'].mean()
platform_likes.plot(kind='bar')
plt.ylabel('Average Likes')
plt.show()                                                                                                                       

# Farklı platformlardaki duygu dağılımı ile bu platformlardaki etkileşim oranlarını karşılaştırmak,
# belirli platformların kullanıcı psikolojisine etkisi olup olmadığını anlamaya yardımcı olur.                                                                                                                    Aşırı Etkileşim Alan İçeriklerin Özellikleri: Aşırı etkileşim alan içerikler kullanıcıların dikkatini daha fazla çekiyor olabilir, bu da sosyal medyada dikkat çekici veya manipülatif içeriklerin nasıl öne çıktığını gösterebilir.
# Günlük Etkileşim Yoğunluğu ve Duygu Durumu Trendleri: Günlük veya haftalık duygu durumu değişimleri,
# kullanıcıların belirli günlerde daha yoğun etkileşim gösterip göstermediğini ve bu durumun içerik 
# türlerine göre nasıl değiştiğini gösterebilir. Aşağıdakiko dizisinde ise bunun analizi yapılmıştır.





# Platformlardaki duygu dağılımını görselleştirme
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Platform', hue='Sentiment')
plt.title("Farklı Platformlardaki Duygu Dağılımı")
# Legend'i grafiğin dışına taşıma ve daha okunaklı hale getirme
plt.legend(title='Sentiment', bbox_to_anchor=(1, 1), loc='upper left', fontsize='small', ncol=3)
plt.subplots_adjust(left=0.05)  # Sol kenar boşluğunu azaltarak grafiği sola kaydırır
plt.tight_layout()  # Grafik elemanlarının sığması için kullanılır
plt.show()




# Platformlara göre etkileşim oranlarını hesaplama ve görselleştirme
df['Engagement'] = df['Likes'] + df['Retweets']
platform_engagement = df.groupby('Platform')['Engagement'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=platform_engagement, x='Platform', y='Engagement')
plt.title("Farklı Platformlardaki Ortalama Etkileşim Oranları")
plt.show()

# Aşırı etkileşim alan içerikleri belirleme (üst %10)
top_10_percent_engagement = df['Engagement'].quantile(0.9)
high_engagement_content = df[df['Engagement'] > top_10_percent_engagement]

# Aşırı etkileşim alan içeriklerin özelliklerini görselleştirme
plt.figure(figsize=(12, 6))
sns.countplot(data=high_engagement_content, x='Sentiment')
plt.title("Aşırı Etkileşim Alan İçeriklerin Duygu Dağılımı")
# Duygu etiketlerini dikey olarak yazma
plt.xticks(rotation=90)  # Etiketleri 90 derece döndürerek dikey yapar
plt.tight_layout()  # Grafik elemanlarının sığması için kullanılır
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=high_engagement_content, x='Platform')
plt.title("Aşırı Etkileşim Alan İçeriklerin Platform Dağılımı")
plt.show()

