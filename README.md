# spam-mail-detection_LSTM
LSTM ALGORİTMASINI KULLANARAK SPAM ELEKTRONİK POSTA TESPİTİ YAPMA VE ELEKTRONİK POSTA SINIFLANDIRMA
Projede ENRON Corpus indirilmiş olup bu veri seti öncelikli kullanılmıştır.
Bu projede kendi gömülümün yanında Google’ın ve Stanford Üniversitesinin Glove embedding’i kullanılmıştır. Bunların her birisi ayrı ayrı eğitilmiştir. Eğitilen bu iki embedding’e göre kendi embedding’imi oluşturarak üç ayrı embedding’in sonuçları kullanıcıya döndürülmüştür. 
Bu projede kullanılan veri setlerini kullanarak bir sınıflandırılma işlemi yapılmıştır. Bu çalışma başka diller eklenerek genişletilebilir ve farklı yönlerle daha kapsamlı bir çalışma haline getirilebilir. 
Başka yöntemler ve başka veri işleme yolları ile tahmin sonuçları değiştirilebilir, geliştirilebilir. İki farklı veri seti ile ve 3 farklı embedding kullanılarak yapılan bu projede LSTM modeli kullanılmıştır. Bu çalışmada kendi oluşturduğum embeddingin Glove ve GoogleNews Vector’den daha yüksek doğrulukla çalışmasının sebebi, veri setlerimi spam ve ham maillerden olacak şekilde seçmem ve buna göre eğitim olması. GlovVe ve Google embeddingleri ise genel dil modelleridir ve daha geniş alanlara hitap etmektedir. Bu yüzden kendi oluşturduğum embedding diğerlerine göre spesifik yerleri, jargon ve ifadeleri daha iyi yakalamaktadır. Aynı zamanda belli bir veri setine göre optimize edilmiş olması, LSTM kullanılmış olması ve diğerlerinden farklı mimaride kullanılmış olması da bu durumda etkilidir. Yani kişisel tercihlere göre oluşturulmuş olması ve buna göre eğitilip, optimize edilmiş olmasından dolayı daha geniş alanlara hitap eden embeddinglere göre doğruluk açısından daha yüksek oranlara ulaşmıştır.
