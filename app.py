import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from PIL import Image
import tempfile
import base64

# -------------------------- 页面配置 --------------------------
st.set_page_config(
    page_title="YOLO 动画角色检测",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- 自定义按钮样式 --------------------------
st.markdown("""
<style>
/* 整体背景色 */
.stApp{
    background:#F5F7FA;  /* 浅灰蓝色，护眼 */
}

/* 侧边栏背景色 */
[data-testid="stSidebar"]{
    background:#E8EEF7;
}

/* 页面标题 */
h1{
    color:#1F2937;
}

/* 自定义按钮 */
div.stButton > button{
    border-radius:50px;
    height:50px;
    width:180px;
    font-size:18px;
    font-weight:bold;
    background-color:#FF8C00;
    color:white;
    border:none;
    margin:5px;
}
div.stButton > button:hover{
    background-color:#FFA500;
    color:#000;
}

/* 图片圆角 */
img{
    border-radius:12px;
}

/* 百科资料卡片样式 */
.animation-card{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0 2px 8px rgba(0,0,0,0.1);
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)
# -------------------------- 加载模型 --------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------- IP Webcam 配置 --------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp;http_verify_certificate;0;allowed_unsafe_types;all;fflags;nobuffer;flags;low_delay"
)

# -------------------------- 动画角色百科资料 --------------------------
animation_info = {
    "Boonie Bears": {
        "title": "熊出没",
        "image": "https://cdn.jsdelivr.net/gh/Huluwa-create/animation-images@main/bear.png",
        "year": "2012",
        "country": "中国",
        "genre": "喜剧、冒险、儿童",
        "intro": "《熊出没》（Boonie Bears），是华强方特（深圳）动漫有限公司推出的三维动画系列作品，刘富源、邢旭辉、叶天龙、林汇达等人担任导演，徐芸、万秦、姜璐等人担任编剧，张伟、张秉君、谭笑等人为剧中角色配音。 [199-200]第一部于2012年1月22日在中央电视台少儿频道首播。 [201]截至2026年4月，该动画共播出23部1300集（主线TV版14部884集、幼教系列7部312集、番外系列2部104集）、电影13部（贺岁片2部，大电影11部）、舞台剧2部、互动剧1部。该系列动画在《熊出没之探险日记》之前，剧情聚焦熊大、熊二兄弟与伐木工光头强在森林保护主题下的幽默对决，传递“自然、环保、健康、快乐”的理念。之后，讲述了熊大、熊二与当导游的光头强为了保护狗熊岭与敌人对抗的搞笑故事 [202]。2013年，该动画因暴力失度、语言粗俗等内容问题被央视点名批评并要求整改。2011年，《熊出没》入选2011年意大利“海湾卡通节”普钦内拉奖、入选意大利Film Festival della Lessinia最佳动漫奖。2013年，《熊出没之环球大冒险》夺得中国动画电影“金猴奖”和中国漫画形象“金猴奖”。2014年，熊出没品牌入选2013年亚洲授权业大奖（The Asian Licensing Awards 2013）“最佳新晋授权品牌”。 [203]2015年，获得韩国釜山国际电影节BIFF Cinekids奖。 [204]2019年，获第四届“玉猴奖”年度十大最具商业价值动漫IP 。 2022年，《熊出没·重返地球》获评第35届中国电影金鸡奖“最佳美术片”奖。2025-2026年度，获评“国家文化出口重点项目” ",
        "characters": ["熊大", "熊二", "光头强"],
        "link": "https://baike.baidu.com/item/熊出没"
    },
    "Black Cat Detective": {
        "title": "黑猫警长",
        "image": "https://cdn.jsdelivr.net/gh/Huluwa-create/animation-images@main/heimao.png",
        "year": "1984",
        "country": "中国",
        "genre": "侦探、教育、儿童",
        "intro": "《黑猫警长》是由上海美术电影制片厂根据诸志祥同名小说改编，戴铁郎、范马迪、熊南清执导的5集动画片。讲述了机智、勇敢、帅气的黑猫警长率领警士痛歼搬仓鼠，破侦螳螂案，消灭一只耳等一个又一个危害森林安全的案件，令森林中的各种动物得以过上安枕无忧的日子的故事。该动画自1984年播出以来，一直受到广大观众，尤其是少年儿童们的喜爱，成为20世纪八九十年代少儿的美好记忆。2022年1月，为迎接第二个中国人民警察节，人民日报新媒体联合小安工作室、上海美术电影制片厂，推出动画片《黑猫警长》国安特别篇。2022年六一儿童节到来之际，优酷推出六一儿童节4K修复主题片单，包括《黑猫警长》。 9月，《动画——黑猫警长》特种邮票一套5枚发行 。",
        "characters": ["黑猫警长", "白猫班长", "一只耳"],
        "link": "https://baike.baidu.com/item/黑猫警长"
    },
    "Tom and Jerry Jerry": {
        "title": "猫和老鼠",
        "image": "https://cdn.jsdelivr.net/gh/Huluwa-create/animation-images@main/tomjerry.png",
        "year": "1940",
        "country": "美国",
        "genre": "喜剧",
        "intro": "《猫和老鼠》（Tom and Jerry）是由威廉·汉纳与约瑟夫·巴伯拉于1939年为美国米高梅电影公司创作的喜剧动画短片，首部剧集《甜蜜的家》于1940年2月10日首播。该片以家猫汤姆与老鼠杰瑞的追逐打闹为主线，通过夸张的肢体喜剧展现两者日常冲突，弱化动物世界的暴力元素。汉纳与巴伯拉在1940至1958年间主导制作了114集短片，其中7集获得奥斯卡最佳动画短片奖。米高梅动画部曾于1957年、1967年两度关闭，期间吉恩·戴奇与查克·琼斯分别接手制作13集和34集。1975年汉纳与巴伯拉重新启动创作，延续经典IP生命力。汤姆的造型从四肢着地逐渐演变为直立拟人化，杰瑞则保持老鼠基本特征。动画于1970年引进中国台湾（译名“妙妙妙”），1980年代进入中国大陆，2017年登陆央视音乐频道。部分剧集因含吸烟镜头在英国播出时遭删减。",
        "characters": ["Tom", "Jerry"],
        "link": "https://baike.baidu.com/item/猫和老鼠"
    },
    "GG Bond": {
        "title": "猪猪侠",
        "image": "https://cdn.jsdelivr.net/gh/Huluwa-create/animation-images@main/heropig.png",
        "year": "2005",
        "country": "中国",
        "genre": "冒险、科幻",
        "intro": "《猪猪侠》是由广东咏声动漫股份有限公司制作的3D动画系列，始创于2005年，该动画由古志斌、陆锦明等人执导，古志斌等人编剧，陆双、祖晴、陈志荣、徐经纬等演员为剧中角色配音。涵盖竞技等题材，TV动画曾多次刷新国产动画收视纪录。该片以猪猪侠勇敢冒险的冒险精神、保卫家园的责任意识、扞卫正义的坚定信念为原点，主要讲述了主人公猪猪侠与伙伴们一起保护童话世界的故事，进行系列创作，传达正义勇敢、坚持不懈等积极正面的精神内涵，传递正能量，力图将其打造成全民儿童英雄 IP。自2005年起，《猪猪侠》每年推出至少1季TV动画、1部院线电影及1部舞台剧，IP授权覆盖玩具、主题乐园等12个产业，截至2024年授权合作企业近百家，IP产品年度收入达8.893亿元，衍生品年零售市值超50亿元。2025年推出20周年纪念电影《猪猪侠·一只老猪的逆袭》，首次尝试全龄化叙事 [24] [31-32]。截至2025年已累计推出18季TV动画、10部院线电影及4部舞台剧 。该系列2009年获“年度最具产业价值动画”奖，2012年摘得“白玉兰奖”国产动画片金奖，2015年入选“五个一工程”优秀作品。2018年《猪猪侠之环球日记》入选年度优秀国产电视动画片，同年，获香港国际授权展“亚洲最佳授权项目大奖”，2019年该系列获“年度十大最具商业价值动漫IP”，2020年入选国家广电总局抗疫公益展播节目。2025年10月，《猪猪侠》系列动画被评为2024年“视听中国全球播映”优秀作品。截至2024年，系列电影累计票房超4亿元，吸引超1200万院线观众。",
        "characters": ["猪猪侠", "菲菲", "超人强", "波比","小呆呆"],
        "link": "https://baike.baidu.com/item/猪猪侠"
    },
    "Calabash Brothers": {
        "title": "葫芦娃",
        "image": "https://cdn.jsdelivr.net/gh/Huluwa-create/animation-images@main/brother.png",
        "year": "1986",
        "country": "中国",
        "genre": "神话、奇幻",
        "intro": "《葫芦兄弟》（又名：葫芦娃），是上海美术电影制片厂出品，胡进庆、葛桂云、周克勤执导，姚忠礼、杨玉良、胡进庆编剧。于1985-1987年原创出品的13集系列剪纸动画片，是中国动画第二个繁荣时期的代表作品之一，已经成为中国动画的经典。该动画片讲述7只神奇的葫芦，7个本领超群的兄弟，为救亲人前赴后继，展开了与妖精们的周旋。《葫芦兄弟》是国内原创经典动画之一 ，该动画自1986年播出以来，一直受到广大观众，尤其是少年儿童们的喜爱。该动画片获得中国广播电影电视部1986—1987年优秀影片奖第三届中国儿童少年电影童牛奖首届中国影视动画节目展播三等奖。 另有续集《葫芦小金刚》，拍摄于约1989——1991年。",
        "characters": ["大娃","二娃","三娃","四娃","五娃","六娃","七娃"],
        "link": "https://baike.baidu.com/item/葫芦兄弟"
    },
    "Peppa Pig": {
        "title": "小猪佩奇",
        "image": "https://cdn.jsdelivr.net/gh/Huluwa-create/animation-images@main/pinkpig.png",
        "year": "2004",
        "country": "英国",
        "genre": "学龄前教育",
        "intro": "《小猪佩奇》（Peppa Pig）是英国动画公司Astley Baker Davies、Entertainment One制作的原创欧洲儿童系列电视动画 [11] [13]，由内维尔·阿斯特利、马克·贝克等编剧，内维尔·阿斯特利、马克·贝克、菲尔·霍尔与乔里斯·范胡尔岑执导 [14]，莫温娜·班克斯、理查德·赖丁斯、蕾拉·法扎德等担任配音。截至2025年，小猪佩奇共播出十季 [26]，合计总集数为354集 [36]。2024年8月3日第十季在中国大陆首播 [26]，2025年8月第十一季正式上线 [61-65]。《小猪佩奇》讲述小猪佩奇是一只非常可爱的粉红猪，她与弟弟乔治、爸爸、妈妈快乐地住在一起，和家人、朋友间发生的日常故事。粉红猪小妹最喜欢做的事情是玩游戏，打扮得漂漂亮亮，度假，以及住在小泥坑里快乐地跳上跳下，她还喜欢到处探险，虽然有些时候会遇到一些小状况，但总可以化险为夷。2022年动画首次引入同性情侣角色，2025年新增家庭成员小猪宝宝伊薇。角色造型通过头部与尾巴变化区分物种，身体形状区分性别，头部轮廓区分年龄。2004年5月31日在英国电视五台首播 ，2005年在美国首播 。2005年获得法国安纳西国际动画节大奖 ，2011年获得BAFTA最佳学前动画片奖 ，2005年、2011年、2012年分别获得英国儿童学会奖 。2015年6月引进中国大陆在中央电视台少儿频道首播。2022年俄罗斯法院裁定允许企业未经授权使用该动画角色。2025年全球首家室内冰雪主题乐园在北京建成，同年宣布2027年在上海崇明长兴岛建成亚洲首个户外主题乐园，并开启电影《小猪佩奇·完美假期》预热宣传。",
        "characters": ["佩奇","乔治","猪爸爸","猪妈妈"],
        "link": "https://baike.baidu.com/item/小猪佩奇"
    },
    "Pleasant Goat and Big Big Wolf": {
        "title": "喜羊羊与灰太狼",
        "image": "https://cdn.jsdelivr.net/gh/Huluwa-create/animation-images@main/goat.png",
        "year": "2005",
        "country": "中国",
        "genre": "喜剧、冒险",
        "intro": "《喜羊羊与灰太狼》问世于2005年，是由广东原创动力文化传播有限公司制作的原创动画作品系列，以友情、搞笑、童话为主题。该动画系列以羊族和狼族之间妙趣横生的故事为主线，讲述了羊狼从斗争到和平的故事。 [1]《喜羊羊与灰太狼》以羊和狼两大族群间妙趣横生的争斗为主线，剧情的轻松诙谐风格，情节爆笑，对白幽默，还巧妙地融入社会中的新鲜名词。这部超强人气的长篇动画以“童趣但不幼稚，启智却不教条”的鲜明特色赢得了很多粉丝，深受广大小朋友的喜爱。2009年9月21日，《喜羊羊与灰太狼》获第十一届精神文明建设“五个一工程”。2010年，《喜羊羊与灰太狼》获最佳收视表现大奖。2015年，《喜羊羊与灰太狼》获第12届中国动漫金龙奖最佳动漫IP奖。2019年，《喜羊羊与灰太狼》获第四届“玉猴奖”年度十大最具商业价值动漫IP 。2025年1月，《喜羊羊与灰太狼》获2024微博之夜线上年度评选“微博年度喜爱动漫”。2026年，《喜羊羊与灰太狼》获「2025年微博ACG次元大赏·2026十大最受期待动漫」。截至2026年2月，《喜羊羊与灰太狼》共播出作品44季3144集（主线32季2465集、网络短剧12季679集）、电影12部（动画电影10部、真人电影2部）、舞台剧5部。《喜羊羊与灰太狼》是中国拥有广泛知名度的本土动漫品牌。它除了有丰富的电视动画剧集，更推出大电影，从电视荧屏走向电影银幕！另外《喜羊羊与灰太狼》在全国率先创立动漫人偶剧团、推出主题杂志与手游等，通过全方位发展，铸就中国优秀动漫品牌。",
        "characters": ["喜羊羊","懒羊羊","沸羊羊","美羊羊","灰太狼","红太狼"],
        "link": "https://baike.baidu.com/item/喜羊羊与灰太狼"
    }
}

# -------------------------- 状态管理 --------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "last_result_img" not in st.session_state:
    st.session_state.last_result_img = None
if "shown_classes" not in st.session_state:
    st.session_state.shown_classes = set()

# -------------------------- 界面 --------------------------
st.title("📹 YOLO 动画角色检测")
st.divider()

# -------------------------- 模式选择 --------------------------
mode = st.radio("选择检测模式", ["ip摄像头检测", "图片检测", "视频检测"])
# -------------------------- 模式切换检测 --------------------------
if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode

if st.session_state.last_mode != mode:
    st.session_state.shown_classes.clear()
    st.session_state.last_mode = mode
# -------------------------- 侧边栏参数 --------------------------
with st.sidebar:
    st.header("⚙️ 参数设置")
    conf_thresh = st.slider("置信度", 0.1, 1.0, 0.3, 0.05)
    iou_thresh = st.slider("IOU", 0.1, 0.9, 0.45, 0.05)
    st.info("低延迟模式")

# -------------------------- 显示动画百科资料 --------------------------
if "shown_classes" not in st.session_state:
    st.session_state.shown_classes = set()

if not isinstance(st.session_state.shown_classes, set):
    st.session_state.shown_classes = set(st.session_state.shown_classes)
def display_animation_info_nonblocking(detected_classes):
    """
    自动把检测到的动画角色信息显示成独立卡片。
    每个角色一个卡片，颜色护眼，带阴影和圆角。
    """
    if "shown_classes" not in st.session_state:
        st.session_state.shown_classes = []

    unique_classes = set(detected_classes)

    new_classes = [
        cls for cls in unique_classes
        if cls in animation_info
           and cls not in st.session_state.shown_classes
    ]

    if not new_classes:
        return

    st.session_state.shown_classes.update(new_classes)

    # 内嵌 CSS 卡片样式
    st.markdown(
        """
        <style>
        .animation-card {
            background:#EAF7EA;  /* 护眼浅绿色 */
            border:1px solid #B7D8B7;   /* 淡绿色边框 */
            border-radius:15px;
            padding:15px;
            margin-bottom:15px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);
        }
        .animation-card img {
            max-width:100%;
            border-radius:10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 遍历每个新检测到的角色，生成卡片
    for cls in new_classes:
        info = animation_info[cls]
        st.markdown(f"""
        <div class="animation-card">
            <h3>{info['title']}</h3>
            <div style="display:flex; gap:15px; flex-wrap:wrap;">
                <img src="{info['image']}" width="200">
                <div>
                    <p><strong>首播时间：</strong>{info['year']}</p>
                    <p><strong>国家/地区：</strong>{info['country']}</p>
                    <p><strong>类型：</strong>{info['genre']}</p>
                    <p><strong>简介：</strong>{info['intro']}</p>
                    <p><strong>主要角色：</strong>{', '.join(info['characters'])}</p>
                    <p><a href="{info['link']}" target="_blank">🔗 查看百科详情</a></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
# -------------------------- 模式1：IP摄像头检测 --------------------------
if mode == "ip摄像头检测":
    st.subheader("🔗 填入 IP Webcam 显示的地址")
    video_url = st.text_input("IP Webcam 地址", value="http://10.27.90.65:8080/video")

    col1, col2 = st.columns(2)
    start = col1.button("▶️ 启动检测")
    stop = col2.button("⏹️ 停止检测")

    frame_placeholder = st.empty()
    status_text = st.empty()
    fps_display = st.empty()

    # 启动摄像头
    if start:
        st.session_state.shown_classes.clear()
        st.session_state.running = True
        status_text.info("正在连接 IP Webcam...")

        cap = cv2.VideoCapture(video_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        st.session_state.cap = cap

        if not cap.isOpened():
            status_text.error("❌ 连接失败！请检查网络/地址/IP Webcam 是否开启")
            st.session_state.running = False

        # 停止摄像头
    if stop:
        st.session_state.running = False
        if st.session_state.cap:
            st.session_state.cap.release()
        frame_placeholder.empty()
        status_text.success("✅ 已停止")
        fps_display.empty()

        # 实时检测循环
    frame_count = 0
    start_time = time.time()

    while st.session_state.running and st.session_state.cap.isOpened():
        cap = st.session_state.cap
        ret, frame = cap.read()

        if not ret:
            status_text.warning("🔄 读取失败，正在重试...")
            time.sleep(0.1)
            continue

        # 每两帧检测一次
        if frame_count % 2 == 0:
            results = model(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
            result_img = results[0].plot()
            st.session_state.last_result_img = result_img

            # 获取检测到的类别名称
            detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        else:
            if st.session_state.last_result_img is not None:
                result_img = st.session_state.last_result_img
            detected_classes = []  # 非检测帧不更新类别

        # 放大画面1.5倍
        height, width = result_img.shape[:2]
        resized_frame = cv2.resize(result_img, (int(width * 1.5), int(height * 1.5)))

        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # 显示FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps_display.metric("实时帧率", f"{frame_count / elapsed:.1f} FPS")

        status_text.success("✅ 正在检测中...")

        # 显示检测到的动画百科资料
        if detected_classes:
            display_animation_info_nonblocking(detected_classes)

    # 释放资源
    if not st.session_state.running and st.session_state.cap:
        st.session_state.cap.release()

# -------------------------- 模式2：图片检测 --------------------------
elif mode == "图片检测":
    st.subheader("🖼️ 上传图片进行YOLO检测")
    img_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        st.session_state.shown_classes.clear()
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)

        results = model(img_np, conf=conf_thresh, iou=iou_thresh)
        res_img = results[0].plot()
        st.image(res_img, caption="检测完成", channels="RGB", width=500 )

        # 获取检测到的类别
        detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        if detected_classes:
            display_animation_info_nonblocking(detected_classes)

# -------------------------- 模式3：视频检测 --------------------------
elif mode == "视频检测":
    st.subheader("🎞️ 上传视频进行YOLO检测")
    video_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:

        st.session_state.shown_classes.clear()

        tfile = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.mp4'
        )
        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("❌ 无法打开视频文件")
            os.unlink(tfile.name)
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 30.0

        st.info(f"📊 视频信息: 总帧数 {total_frames}, 原始帧率 {original_fps:.1f} FPS")

        progress_bar = st.progress(0)
        stframe = st.empty()
        status = st.empty()
        fps_display = st.empty()

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_thresh, iou=iou_thresh)
            res_frame = results[0].plot()
            rgb = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb, channels="RGB", width=800)

            frame_count += 1
            elapsed_time = time.time() - start_time

            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(progress)

            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_display.metric("处理帧率", f"{current_fps:.1f} FPS")

            status.info(f"▶️ 正在检测... {frame_count}/{total_frames} 帧 ({progress * 100:.1f}%)")

            detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
            if detected_classes:
                display_animation_info_nonblocking(detected_classes)

            target_frame_time = 1.0 / original_fps
            actual_elapsed = time.time() - start_time
            expected_time = frame_count * target_frame_time
            sleep_time = expected_time - actual_elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        total_elapsed = time.time() - start_time
        os.unlink(tfile.name)

        status.success(f"✅ 视频检测完成! 共处理 {frame_count} 帧, 耗时 {total_elapsed:.1f} 秒")
        fps_display.metric("平均处理帧率", f"{frame_count / total_elapsed:.1f} FPS")