"""Test genre classification with Gen Z text"""
from nlp.genre_classifier import genre_classifier

# Test with Gen Z essay text (excerpt)
gen_z_text = """
لم يعد الجيل الجديد يقاس بالعمر وحده، بل يقاس بالبيئة الرقمية التي ولد فيها.
جيل Z هو أول جيل ولد والإنترنت قائم، والهواتف الذكية متاحة، والمنصات حاضرة في تفاصيل الحياة اليومية.
مصطلح جيل Z يشير في الغالب إلى مواليد منتصف التسعينيات حتى عام 2010 تقريباً، 
وقد ولد المفهوم في بيئات التسويق وشركات التكنولوجيا لفهم المستهلك الجديد، 
ثم انتقل إلى علم الاجتماع، وعلم النفس.
هذا الجيل يختلف جذرياً عمن سبقه، في اللغة، وفي مصادر المعرفة، وفي علاقته بالسلطة.
الهوية عنده أقل صلابة، وأكثر قابلية للتبدل، والانتماء الرقمي أكثر حضوراً من الانتماء المكاني.
تأثيره السياسي عميق، لأنه جيل لا يقبل الوصاية بسهولة.
ثقافياً، هو جيل الصورة قبل النص، والفيديو قبل المقال.
اجتماعياً، تغيرت أنماط العلاقات.
"""

print("=" * 60)
print("Testing Gen Z Text Classification")
print("=" * 60)

result = genre_classifier.classify(gen_z_text)
print(f"\n🏷️ GENRE: {result['genre']}")
print(f"📊 CONFIDENCE: {result['confidence']}")
print(f"\n📈 SCORES:")
for genre, score in result['all_scores'].items():
    print(f"   {genre}: {score}")
print(f"\n🔑 MATCHED KEYWORDS: {result['matched_keywords']}")

# Show all matched educational keywords
edu_matches = [kw for kw in genre_classifier.EDUCATIONAL_KEYWORDS if kw in gen_z_text]
print(f"\n📚 All Educational Keywords Found ({len(edu_matches)}):")
print(f"   {edu_matches}")
