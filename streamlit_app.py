import io

import streamlit as st
import PIL as pil
from ucall.client import Client

st.set_page_config(
    page_title="USearch Images",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("USearch Images")

ip_address: str = st.secrets["SERVER_IP"]
ip_address = "0.0.0.0"
ip_address = None


@st.cache_resource
def get_server():
    import server

    return server


def unwrap_response(resp):
    if ip_address is None:
        return resp
    else:
        return resp.json


# Starting a new connection every time seems like a better option for now
# @st.cache_resource
# def get_client() -> Client:
#     return Client(uri=ip_address)
client = Client(uri=ip_address) if ip_address is not None else get_server()

image_query = bytes()
text_query = str()
results = list()


examples = {
    "🇺🇸": [
        "exotic fruits in punchy colors",
        "neon signs at a gas station",
        "a girl wandering alone in the forest",
        "camping in a desert under the stars",
        "global warming protests",
        "clownfish hiding in corals",
        "birds flying close to the water",
    ],
    "🇨🇳": [
        "色彩鲜艳的异国水果",
        "加油站的霓虹灯",
        "一个女孩独自在森林里徘徊",
        "在星空下的沙漠中露营",
        "全球变暖抗议",
        "躲在珊瑚里的小丑鱼",
        "鸟儿飞近水面",
    ],
    "🇮🇳": [
        "तीखे रंगों में विदेशी फल",
        "एक गैस स्टेशन पर नियॉन संकेत",
        "एक लड़की जंगल में अकेली घूम रही थी",
        "तारों के नीचे रेगिस्तान में डेरा डालना",
        "ग्लोबल वार्मिंग का विरोध",
        "मूंगों में छिपी क्लाउनफ़िश",
        "पानी के करीब उड़ते पक्षी",
    ],
    "🇩🇪": [
        "exotische Früchte in kräftigen Farben",
        "Leuchtreklamen an einer Tankstelle",
        "ein Mädchen, das allein im Wald wandert",
        "Camping in einer Wüste unter den Sternen",
        "Proteste gegen die globale Erwärmung",
        "Clownfische verstecken sich in Korallen",
        "Vögel fliegen in der Nähe des Wassers",
    ],
    "🇦🇲": [
        "էկզոտիկ մրգեր թափանցիկ գույներով",
        "նեոնային ցուցանակներ բենզալցակայանում",
        "անտառում մենակ թափառող աղջիկ",
        "ճամբարում աստղերի տակ գտնվող անապատում",
        "գլոբալ տաքացման դեմ բողոքի ցույցեր",
        "ծաղրածու ձուկ թաքնված մարջաններում",
        "թռչուններ, որոնք թռչում են ջրի մոտ",
    ],
    "🇫🇷": [
        "fruits exotiques aux couleurs acidulées",
        "enseignes au néon dans une station-service",
        "une fille errante seule dans la forêt",
        "camper dans un désert sous les étoiles",
        "manifestations contre le réchauffement climatique",
        "poisson-clown caché dans les coraux",
        "oiseaux volant près de l'eau",
    ],
    "🇪🇸": [
        "frutas exóticas en colores llamativos",
        "letreros de neón en una gasolinera",
        "una niña vagando sola en el bosque",
        "acampar en un desierto bajo las estrellas",
        "protestas por el calentamiento global",
        "pez payaso escondido en los corales",
        "pájaros volando cerca del agua",
    ],
    "🇵🇹": [
        "frutas exóticas em cores vibrantes",
        "sinais de néon em um posto de gasolina",
        "uma garota vagando sozinha na floresta",
        "acampar em um deserto sob as estrelas",
        "protestos aquecimento global",
        "peixe-palhaço escondido em corais",
        "pássaros voando perto da água",
    ],
    "🇮🇹": [
        "frutti esotici in colori vivaci",
        "insegne al neon in una stazione di servizio",
        "una ragazza che vaga da sola nella foresta",
        "campeggio in un deserto sotto le stelle",
        "proteste contro il riscaldamento globale",
        "pesce pagliaccio nascosto nei coralli",
        "uccelli che volano vicino all'acqua",
    ],
    "🇵🇱": [
        "egzotyczne owoce w wyrazistych kolorach",
        "neony na stacji benzynowej",
        "dziewczyna wędrująca samotnie po lesie",
        "biwakowanie na pustyni pod gwiazdami",
        "protesty w sprawie globalnego ocieplenia",
        "błazenki ukrywające się w koralowcach",
        "ptaki latające blisko wody",
    ],
    "🇺🇦": [
        "екзотичні фрукти яскравих кольорів",
        "неонові вивіски на АЗС",
        "дівчина блукає сама по лісі",
        "кемпінг у пустелі під зірками",
        "протести проти глобального потепління",
        "риба-клоун ховається в коралах",
        "птахів, що летять близько до води",
    ],
    "🇷🇺": [
        "экзотические фрукты ярких цветов",
        "неоновые вывески на заправке",
        "девушка бродит одна по лесу",
        "кемпинг в пустыне под звездами",
        "протесты против глобального потепления",
        "рыба-клоун прячется в кораллах",
        "птицы летят близко к воде",
    ],
    "🇹🇷": [
        "keskin renklerde egzotik meyveler",
        "bir benzin istasyonunda neon tabelalar",
        "ormanda tek başına dolaşan bir kız",
        "yıldızların altında bir çölde kamp yapmak",
        "küresel ısınma protestoları",
        "mercanlarda saklanan palyaço balığı",
        "suya yakın uçan kuşlar",
    ],
    "🇮🇷": [
        "میوه های عجیب و غریب در رنگ های تند",
        "تابلوهای نئونی در پمپ بنزین",
        "دختری که تنها در جنگل سرگردان است",
        "چادر زدن در بیابان زیر ستاره ها",
        "اعتراضات گرمایش زمین",
        "دلقک ماهی که در مرجان ها پنهان شده است",
        "پرندگانی که نزدیک آب پرواز می کنند",
    ],
    "🇮🇱": [
        "פירות אקזוטיים בצבעים נוקבים",
        "שלטי ניאון בתחנת דלק",
        "ילדה משוטטת לבד ביער",
        "קמפינג במדבר מתחת לכוכבים",
        "מחאות התחממות כדור הארץ",
        "דגי ליצנים מתחבאים באלמוגים",
        "ציפורים עפות קרוב למים",
    ],
    "🇸🇦": [
        "فواكه غريبة بألوان متقنة",
        "إشارات النيون في محطة وقود",
        "فتاة تتجول وحدها في الغابة",
        "التخييم في صحراء تحت النجوم",
        "احتجاجات الاحتباس الحراري",
        "كلوونفيش يختبئ في الشعاب المرجانية",
        "الطيور تحلق بالقرب من الماء",
    ],
    "🇻🇳": [
        "trái cây kỳ lạ trong màu sắc punchy",
        "bảng hiệu neon tại một trạm xăng",
        "một cô gái lang thang một mình trong rừng",
        "cắm trại trên sa mạc dưới những vì sao",
        "cuộc biểu tình nóng lên toàn cầu",
        "cá hề trốn trong san hô",
        "chim bay gần mặt nước",
    ],
    "🇹🇭": [
        "ผลไม้แปลกใหม่ในสีที่กัด",
        "ป้ายไฟนีออนที่ปั๊มน้ำมัน",
        "หญิงสาวพเนจรคนเดียวในป่า",
        "ตั้งแคมป์ในทะเลทรายใต้แสงดาว",
        "การประท้วงภาวะโลกร้อน",
        "ปลาการ์ตูนซ่อนตัวอยู่ในปะการัง",
        "นกที่บินอยู่ใกล้น้ำ",
    ],
    "🇮🇩": [
        "buah-buahan eksotis dengan warna mencolok",
        "lampu neon di pom bensin",
        "seorang gadis berkeliaran sendirian di hutan",
        "berkemah di padang pasir di bawah bintang-bintang",
        "protes pemanasan global",
        "ikan badut bersembunyi di karang",
        "burung terbang dekat dengan air",
    ],
    "🇰🇷": [
        "강렬한 색상의 이국적인 과일",
        "주유소의 네온사인",
        "숲에서 혼자 방황하는 소녀",
        "별빛 아래 사막에서 캠핑",
        "지구 온난화 시위",
        "산호에 숨어있는 흰 동가리",
        "물 가까이 날아가는 새들",
    ],
}

text_query: str = st.text_input(
    "Search Bar",
    placeholder="USearch for Images in the Unsplash dataset",
    value=examples["🇺🇸"][0],
    key="text_query",
    label_visibility="collapsed",
)

image_query: io.BytesIO = st.file_uploader("Alternatively, choose an image file")
selected_language = st.radio(
    "Or one of the examples",
    list(examples.keys()),
    horizontal=True,
)
for example in examples[selected_language]:
    if st.button(example):
        text_query = example

columns: int = st.sidebar.slider("Grid Columns", min_value=1, max_value=10, value=8)
max_results: int = st.sidebar.number_input(
    "Max Matches",
    min_value=1,
    max_value=None,
    value=100,
)
dataset_name: str = st.sidebar.selectbox("Dataset", ("unsplash25k",))
size: int = unwrap_response(client.size(dataset_name))


# Search Content

with st.spinner(f"We are searching through {size:,} entries"):
    if image_query:
        image_query = pil.Image.open(image_query).resize((224, 224))
        results = unwrap_response(
            client.find_with_image(
                dataset=dataset_name,
                query=image_query,
                count=max_results,
            )
        )
    else:
        results = unwrap_response(
            client.find_with_text(
                dataset=dataset_name,
                query=text_query,
                count=max_results,
            )
        )


st.success(
    f"Displaying {len(results):,} closest results from {size:,} entries!\nUses [UForm](https://github.com/unum-cloud/uform) AI model, [USearch](https://github.com/unum-cloud/usearch) vector search engine, and [UCall](https://github.com/unum-cloud/ucall) for remote procedure calls.",
    icon="✅",
)


# Visualize Matches

for match_idx, match in enumerate(results):
    col_idx = match_idx % columns
    if col_idx == 0:
        st.write("---")
        cols = st.columns(columns, gap="large")

    with cols[col_idx]:
        st.image(match, use_column_width="always")
