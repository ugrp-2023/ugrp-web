# 기존 코드와 상이한 점 : panel 생성과 관련된 내용을 전부 제거, chatbot 구현에만 필요한 내용들 넣음
#                       아래 코드를 바탕으로 웹이나 앱에 변수 연결이 중요함 (단, 웹과 앱의 연결 방식은 다름)

# 전체 시스템과 에러 관여
import sys
import warnings

# 음성 파일 생성과 재생에 관여
import pygame
import speech_recognition as sr
import time
from gtts import gTTS
import os

# openai와 chatgpt langchain 파일 (라이브러리 파일 다운로드 필요할 수 있음)
import openai
import datetime
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

# openai.api key 연결
sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = ""
OPENAI_API_KEY  = openai.api_key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 오류 났을 경우 무시
warnings.filterwarnings('ignore')

# 전역 변수 생성 - 음성 파일 생성과 삭제에 관여
i=0

# gpt 시리즈 적용
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

# 아래 3개 - 음성 파일 녹음, 생성, 삭제와 관련 - 웹 적용 단계에서 오류가 있으므로 아직 구현하지 말거나, 사용자에 맞는 파일 경로 지정하는 방안 찾아내야 함
# 음성파일 생성 함수
def speak(text):
    tts = gTTS(text=text, lang='ko')
    global i
    filename = "voice" + str(i) + ".mp3"
    tts.save(filename)

    time.sleep(1)
    pygame.mixer.init()
    pygame.mixer.music.load("voice"+str(i)+".mp3")
    pygame.mixer.music.play()

    i += 1
    return

# 음성파일 삭제 함수
def delete():
    n = 0
    file_path = "/" + "voice" + str(n) + ".mp3"

    while os.path.isfile(file_path) == True:
        os.remove(file_path)
        n = n + 1
        file_path = "/" + "voice" + str(n) + ".mp3"

# 음성 녹음 함수
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        said=""
        print("지금 말씀해주세요")
        time.sleep(0.5)
        audio = r.listen(source)

        try:
            said = r.recognize_google(audio, language="ko-KR")
            print("말씀하신 내용입니다:", said)
        except Exception as e:
            print("Exception: " + str(e))

    return said


# Chat GPT와 관련된 함수들 등장

# 데이터 수집 함수
def load_db(files, k):
    all_docs = []
    for file in files: # files 배열 내에 있는 파일들 하나하나 읽어옴
        loader = PyPDFLoader(file) # pdf 파일 읽음
        documents = loader.load() # 문서화
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # 텍스트 분할
        docs = text_splitter.split_documents(documents) # 분할된 텍스트를 docs 배열 안에 넣기
        all_docs.extend(docs) # docs 배열 안에 분할된 텍스트 계속해서 들어오도록 연결

    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(all_docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # 데이터베이스에 임베딩해서 넣기 (데이터의 벡터화)

    return retriever

def qanda(retriever, chain_type):
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    # 임베딩된 데이터를 이용한 대화형 체인 준비
    return qa

# 질문 봇 클래스 생성
class QA_bot:

    # 클래스 내에서 사용할 것들 정의 (파일 경로, qa chain, 대화 기억, 데이터베이스에 저장된 질문, 데이터베이스에 저장된 답변, 표현될 답변 등)
    def __init__(self, *file_paths):
        self.loaded_file = load_db(file_paths, 4)
        self.qa = qanda(self.loaded_file,"stuff")
        self.chat_history = []
        self.db_query = ""
        self.db_response = ""
        self.answer = ""
        self.context = [{'role':'system', 'content':"""
                            이용자들은 재난 대응에 대한 정보와 위험 상황에 대한 안내, 행동요령, 대처방안 등을 찾고 있습니다.\
                            너의 역할은 재난 전문가로서 너가 알고 있는 재난 지식을 통해 정확한 정보와 생존 전략을 사용자 친화적인 방식으로 제공하는 것입니다.\


                            너가 알고 있는 재난 지식 및 재난 상황은 자연재난, 사회재난, 생활안전, 비상대비으로 총 4가지입니다. \
                            자연재난은 침수, 태풍, 호우, 낙뢰, 강풍, 풍랑, 대설, 한파, 폭염, 황사, 지진, 해일, 지진해일, 화산폭발, 가뭄, 홍수, 해수면상승, 산사태, 자연우주물체추락, 우주전파재난, 조류대발생, 적조에 관한 내용입니다.\
                            사회재난은 화재, 산불, 건축물붕괴, 폭발, 교통사고, 전기 및 가스사고, 철도 및 지하철 사고, 유·도선 사고, 해양 선박사고, 식용수, 원전사고, 공동구재난, 대규모수질오염, 가축질병, 댐붕괴, 정전 및 전력부족, 금융전산, 감염병예방, 해양오염사고, 화학물질사고, 항공기사고, 인공우주물체추락, 미세먼지, 정보통신사고, GPS전파혼신재난, 보건의료재난, 사업장대규모인적사고, 공연장 안전, 도로터널사고, 경기장 안전에 관한 내용입니다.\
                            생활안전은 여름철물놀이, 산행안전사고, 응급처치, 해파리피해, 심폐소생술, 붉은불개미, 승강기 안전사고, 어린이 놀이시설 안전사고, 식중독, 실종유괴예방, 학교폭력예방, 가정폭력예방, 억류 및 납치, 석유제품 사고에 관한 내용입니다.\
                            비상대비는 테러, 비상사태, 민방공 경보, 화생방, 비상대비물자준비에 관한 내용입니다.\

                            너가 알고 있는 여러 재난 상황을 위험성, 긴급성, 범위성에 따라 1등급에서 4등급으로 분류할 수 있습니다.\
                            1등급에는 자연재난 중 지진, 지진해일, 화산폭발이 해당되고 사회재난 중 원전사고, 화학물질사고가 해당되고 비상대비 중 테러, 비상사태, 민방공 경보, 화생방, 비상대비물자가 해당됩니다.\
                            2등급에는 자연재난 중 침수, 태풍, 호우, 해일, 홍수, 산사태가 해당되고 사회재난 중 화재, 산불, 건축물붕괴, 폭발, 전기 및 가스사고, 식용수, 댐붕괴, 감염병예방, 도로터널사고가 해당됩니다.\
                            3등급에는 자연재난 중 낙뢰, 강풍, 풍랑, 대설, 한파, 폭염, 황사, 가뭄, 해수면상승이 해당되고 사회재난 내 교통사고, 철도 및 지하철 사고, 유·도선 사고, 해양 선박사고, 공동구재난, 대규모수질오염, 가축질병, 정전 및 전력부족, 금융전산, 해양오염사고, 항공기사고, 미세먼지, 보건의료재난, 사업장대규모인적사고, 공연장 안전, 경기장 안전이 해당됩니다.\
                            4등급에는 자연재난 중 자연우주물체추락, 우주전파재난, 조류대발생, 적조가 해당되고 사회재난 중 인공우주물체추락, 정보통신사고, GPS전파혼신재난이 해당되고 생활안전 전체가 해당됩니다.\


                            너가 수행해야 하는 행동은 4단계로 이루어져 있으니 반드시 단계별로 행동해야 합니다.\

                            1단계: 이용자의 현재 질문의 핵심 내용과 키워드를 빠르게 감지하고 이해합니다.\
                            이해한 질문의 핵심 내용과 키워드가 너가 알고 있는 재난 상황과 관련성이 있는지 판단합니다. 또한, 너가 알고 있는 재난 지식을 통해 질문에 답변할 수 있는지 판단합니다.\
                            이때, 과거의 대화 맥락을 파악하면서 질문을 이해하는 것이 중요합니다.\

                            2단계: 1단계에서 재난과 관련성이 높거나 너의 지식을 통해 답변 가능하다면 너가 알고 있는 재난 지식을 무조건 활용하여 답변을 생성합니다. 추가적으로 재난 상황의 등급을 반드시 파악합니다.\
                            하지만 1단계에서 재난과 관련성이 없거나 낮은 경우 이용자에게 학습되지 않은 재난 상황임을 언급합니다. 그 후 신뢰할 수 있는 출처에서 인터넷을 검색하고 검색해 찾은 내용을 바탕으로 답변을 생성합니다.\

                            3단계: 2단계에서 생성한 답변을 등급에 따라 요약합니다.\
                            1등급에 해당하는 재난인 경우 핵심 내용과 필수적인 내용만 매우 간략하고 명시적으로 요약합니다.\
                            2등급에 해당하는 재난인 경우 핵심 내용과 필수적인 내용만 요약합니다.\
                            3등급과 4등급에 해당하는 재난인 경우 요약을 거의 하지 않습니다.\
                            추가적으로 사용자가 이야기하는 질문 내 재난 상황이 구체적일수록, 그 상황에 맞는 답변만 제공합니다.\

                            4단계: 3단계에서 요약한 내용을 아래 체크리스트에 맞게 확인하고, 최종적으로 답변을 제공합니다.\

                            확인할 때 고려해야 할 사항은 다음과 같습니다. \
                            1. 너무 기초적인 지식으로 글이 길어지지 않았는지 \
                            2. 질문자가 이 글을 읽기 쉬운지 (가독성이 좋은지) \
                            3. 질문에 맞는 답변을 진행했는지 (관계성이 높은지) \
                            4. 구체적인 재난 상황에 알맞은 답변만을 진행했는지\

                            만약 제공될 답변이 위 고려해야 할 사항을 충족하지 못할 경우, 최대 3번까지 1단계부터 다시 시행하도록 합니다.\

                            추가적인 참고사항은 다음과 같습니다.\
                            기술적 또는 사전과 같은 톤을 피하면서 가독성을 위해 노력합니다.\
                            사용자의 이해도를 높이기 위해 유창한 입담으로 답변을 주입합니다.\
                            응답에서 따뜻하고 서비스 지향적인 한국어 톤을 일관되게 유지하세요.\
                            """}]

    # 대화형 체인 작동
    def convchain(self, query):
        result = self.qa({"question": query, "context": self.context, "chat_history": self.chat_history}) # 질문과 기억된 대화 내용을 바탕으로 답변한 내용
        self.chat_history.extend([(query, result["answer"])]) # 질문과 답변은 대화 기억에 저장
        self.db_query = result["generated_question"] # 저장될 질문 부분
        self.db_response = result["source_documents"] # pdf에서 비슷한 내용을 찾아내는 과정
        self.answer = result['answer'] # 웹 상으로 표현될 답변 부분
        return self.answer

# pdf 파일을 여러개 넣을 수 있음 : 배열을 통해 pdf 파일 하나하나를 다 데이터베이스화 함 (load_db 함수에서 확인 가능)
file_paths =[".pdf"]

# QA_bot 클래스 생성
bot = QA_bot(*file_paths)

# 채팅 함수 - flask에서만 사용할 함수들 간단하게 묶어놓은 상태
# (flask - import 이 파이썬 파일 as 내가 사용할 함수명)
# ex> import juseok as cb
#     cb.doing(question)
# 위와 같이 사용하도록 했음
def doing(user_input):

    if user_input.lower() == "speak": # 질문에 speak라는 단어가 들어가 있으면, 음성 녹화 함수 적용 및 질문에 대한 답변 제공
        while True:
            user_input = get_audio()
            answer = bot.convchain(user_input)
            #speak(answer)

    else :
        answer = bot.convchain(user_input) # input이 일반적인 질문일 경우, 질문에 대한 답변 제공
        #speak(response)

    return answer