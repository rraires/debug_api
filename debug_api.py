###  _v7_2_res

import streamlit as st
import requests
import time
import paho.mqtt.client as mqtt
import json
import cv2
import paho.mqtt.client as mqtt
import traceback
# from icecream import ic
from collections import deque
from pprint import pprint
import sys
import numpy as np

st.set_page_config(layout='wide')

MQTT_TOPIC = 'openpose'
BROKER = "mqtt.anglis.com.br"

IMG_BACKGROUND = './roberto.jpeg'

frames = [] #fila com os pontos vindos do mqtt

flag_connected = 0

resolucao = (960,540)
# resolucao = (1920,1080)
thickness_=6

# if (int(sys.argv[1])==76712861):
#     IMG_BACKGROUND = './roberto_lab.jpg'
# elif (int(sys.argv[1])==86005480):
#     IMG_BACKGROUND = './quarto_eric.JPG'    
# elif (int(sys.argv[1])==13847801):
#     IMG_BACKGROUND = './quarto_eric.JPG'   
# elif (int(sys.argv[1])==18765083):
#     IMG_BACKGROUND = './inova2.png'    
# else:
#     IMG_BACKGROUND = './inova2.png'  

FRAMES_PREDICT = 30 #quantos quadros para predict na rede
# 1. New detection variables
sequence = deque(maxlen=FRAMES_PREDICT) #define o tamanho maximo da lista para o predict, de acordo com a constante FRAMES_PREDICT
# predictions = []
# threshold = 0.5
face_lines = [(270, 409), (176, 149), (37, 0), (84, 17), (318, 324), (293, 334), (386, 385), (7, 163), (33, 246), (17, 314), (374, 380), (251, 389), (390, 373), (267, 269), (295, 285), 
(389, 356), (173, 133), (33, 7), (377, 152), (158, 157), (405, 321), (54, 103), (263, 466), (324, 308), (67, 109), (409, 291), (157, 173), (454, 323), (388, 387), (78, 191), (148, 176), (311, 310), (39, 37), (249, 390), (144, 145), (402, 318), (80, 81), (310, 415), (153, 154), (384, 398), (397, 365), (234, 127), (103, 67), (282, 295), (338, 297), (378, 400), (127, 162), (321, 375), (375, 291), (317, 402), (81, 82), (154, 155), (91, 181), (334, 296), (297, 332), (269, 270), (150, 136), (109, 10), (356, 454), (58, 132), (312, 311), (152, 148), (415, 308), (161, 160), (296, 336), (65, 55), (61, 146), (78, 95), (380, 381), (398, 362), (361, 288), (246, 161), (162, 21), (0, 267), (82, 13), (132, 93), (314, 405), (10, 338), (178, 87), (387, 386), (381, 382), (70, 63), (61, 185), (14, 317), (105, 66), (300, 293), (382, 362), (88, 178), (185, 40), (46, 53), (284, 251), (400, 377), (136, 172), (323, 361), (13, 312), (21, 54), (172, 58), (373, 374), (163, 144), (276, 283), (53, 52), (365, 379), (379, 378), (146, 91), (263, 249), (283, 282), (87, 14), (145, 153), (155, 133), (93, 234), (66, 107), (95, 88), (159, 158), (52, 65), (332, 284), (40, 39), (191, 80), (63, 105), (181, 84), (466, 388), (149, 150), (288, 397), (160, 159), (385, 384)]

hands_lines = [(3, 4), (0, 5), (17, 18), (0, 17), (13, 14), (13, 17), (18, 19), (5, 6), (5, 9), (14, 15), (0, 1), (9, 10), (1, 2), (9, 13), (10, 11), (19, 20), (6, 7), (15, 16), (2, 3), (11, 12), (7, 8)]

# pose_lines = [(15, 21), (16, 20), (13, 15), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3), (11, 12), (27, 29)]
pose_lines = [(15, 21),(15, 17),(15, 19),(17, 19), (13, 15),(11, 13),  (23, 11), (11, 12),(12,24),(23, 24), (12, 14), (14, 16), (16, 22),(16, 20),(16, 18),(18, 20), (25, 27),(23, 25), (27, 29),(27, 31),(29, 31),(24, 26),(26, 28),(30, 32), (28, 32),(28, 30),(9, 10),(3, 7),  (6, 8) ,  (4, 5), (5, 6),   (0, 1),  (1, 2), (0, 4),    (2, 3) ]

pose_lines_coloar = [
    #(B,G,R)
    #Mão Esquerda
    (11,255,45),  #ombro esquerdo [1,5]
    (11,255,45),  #ombro direito [2,1]
    (11,255,45),    #braço esquerdo [5,6]
    (11,255,45),

    #Braço Esquerda
    (255,47,111),    #braço direito [2,3]
    (255,47,111),  #antebraço esquerdo [6,7]


    #tronco
    (255,90,23),  #antebraço direito [3,4]
    (255,12,255),  #pescoço / queixo [1,0]
    (88,255,190),  # tronco [1,8]
    (192,24,100),  # bacia direita [8,9]

    #Braço Direito
    (0,199,122),  #bacia esquerda [8,12]
    (0,199,122),  #coxa esquerda [12,13]


    #Mão Direita
    (255,255,182),  #coxa esquerda [12,13]
    (255,255,182),  #coxa esquerda [12,13]
    (255,255,182),
    (255,255,182),


    #perna Esquerda
    (255,0,0),
    (255,0,0),


    #pe Esquerda
    (0,123,33),
    (0,123,33),
    (0,123,33),

    #perna Direita
    (255,0,255), 
    (255,0,255), 
    #pe Direita
    (200,23,45),
    (200,23,45),
    (200,23,45),
    
    #boca
    (255,252,255),

    #olhos
    (123,2,55),
    (123,2,55),
    (123,2,55),
    (123,2,55),
    (123,2,55),
    (123,2,55),
    (123,2,55),
    (123,2,55)]

POSEBody_PAIRS = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [18, 1], [18, 6], [18, 7], [18, 12], [18, 13]]


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    global flag_connected
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_TOPIC)
    flag_connected = 1

def on_disconnect(client, userdata, rc):
   global flag_connected
   print("disconnected with result code "+str(rc))
   flag_connected = 0    

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global frames
    mqtt_msg = str(msg.payload)
    # if(msg.topic == MQTT_TOPIC and mqtt_msg.find("mmwave") == -1): #procura pela string mmwave nas mensagens para somente essas mensagens
    #     frames.append(msg.payload) #cria uma fila com os pontos
    frames.append(msg.payload) #cria uma fila com os pontos
    

# def on_message(client, userdata, msg):
#     global mqtt_keypoints
#     global mqtt_uuid
#     global obj
#     try:
#         msg = json.loads(msg.payload.decode("utf-8"))
#         dic1 = msg
#         for key, value in dic1.items(): 
#             if (key=="device_id"):
#                 if(str(sys.argv[1]) == str(value)):
                     
#                      for key2, value2 in dic1.items(): 
#                         # print(value)
#                         if (key2=="mediapipe"):
#                             mqtt_keypoints.append(value2)
#                         if (key2=="uuid_pose"):
#                             mqtt_uuid.append(value2)
#                         # if (key2=="obj"):
#                         #     obj.append(value2) 
#     except:
#         print(traceback.print_exc())




def dados_iniciais():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        url = st.radio('Fonte', ['http://app.anglis.com.br:8081/monit', 'http://127.0.0.1:8081/monit'])
        # url = st.radio('Fonte', ['http://127.0.0.1:8081/monit'])
    try:
        response = requests.get(url)
    except:
        st.write('Erro API')
        st.stop()
    data_api = response.json()
    franquias = [record.get("Franquia") for record in data_api]
    # print(data_api)
    with col2:
        selecao_franquia = st.selectbox('Franquia', franquias, key='franquia')
    dados_residentes_franquia = [record.get("Residentes") for record in data_api if record.get("Franquia") == selecao_franquia]
    residentes_franquia = [record.get("Residente") for record in dados_residentes_franquia[0]]
    with col3:
        selecao_residente = st.selectbox('Residente', residentes_franquia, key='residente')
    dados_residente = [record for record in dados_residentes_franquia[0] if record.get("Residente") == selecao_residente]
    comodos_residente = [record.get("comodo") for record in dados_residente[0]['comodos']]
    # comodos_residente_local = [record.get("local") for record in dados_residente[0]['comodos']]
    
    comodos_residente = [record.get("local") for record in dados_residente[0]['comodos']]
    # print(comodos_residente,comodos_residente_local)
    with col4:
        # selecao_comodo = st.selectbox('Cômodo', comodos_residente, key='local')
        selecao_comodo = st.selectbox('Cômodo', comodos_residente, key='comodo')
    return url, selecao_franquia, selecao_residente, selecao_comodo




client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

client.connect(BROKER, 1883, 60)
client.loop_start()  

olho = [93, 94]
img_size = (480,640)


loc =0


# lista_obj_id =[[[], [], [], []], [[], [], [], [], [], [], [111111,222222]], [[], [], [], []], [[], [], [333333, 252525]], [[], [], []], [[], [], [], [], [], [], [], [], []], [[444444,898989,877474], []], [[], [], []]]
# lista_obj =[[[], [], [], []], [[], [], [], [], [], [], ['Cama','Sofa']], [[], [], [], []], [[], [], ['Restrita', 'Bancada']], [[], [], []], [[], [], [], [], [], [], [], [], []], [['Restrita', "Corredor", "Porta"], []], [[], [], []]]
# lista_obj_pos = [[[], [], [], []], [[], [], [], [], [], [], [[0.5234375,0.194444444,0.911458333,0.972222222],[0.000416667,0.694444444,0.286666667,0.007222222]]], [[], [], [], []], [[], [], [[0.005208333,0.046296296,0.364583333,0.972222222],[0.485208333,0.006296296,0.864583333,0.38222222]]], [[], [], []], [[], [], [], [], [], [], [], [], []], [[[0.005208333,0.046296296,0.364583333,0.972222222],[0.364208333,0.276296296,0.904583333,0.972222222],[0.36408333,0.006296296,0.6064583333,0.272222222]], []], [[], [], []]]


# lista_residentes =['Eric - Avós', 'Eric Casa Teste', 'JULIO AMERICO GONZALEZ', 'Roberto Martins', 'Eric - Pais', 'Joaquim Aires Martins', 'Inovabra', 'Brooklin']
# lista_id_residentes = [25249854, 52825284, 32668106, 94852179, 46664908, 16991016, 56081094, 81764839]
# lista_comodos_residentes = [['Oficina', 'Lavanderia', 'Sala', 'Cozinha'], ['Sala mmwave', 'Quarto mmwave', 'Sala', 'Teto', 'Banheiro', 'Sala2 mmwave', 'Quarto'], ['Quintal', 'Sala', 'Escritório', 'Banheiro'], ['Banheiro', 'Teste mmWave', 'Quarto'], ['Escritorio', 'Rua', 'base'], ['Garagem', 'Quarto-Casal', 'Quarto-Hospedes', 'Banheiro-Casal', 'Banheiro-Hospedes', 'Corredor', 'Sala', 'Cozinha', 'Quintal-Fundos'], ['Quarto', 'Banheiro'], ['Quarto', 'Sala Jantar', 'Sala TV']]
# lista_id_comodos = [[22193645, 66480578, 47752558, 15422247], [30905914, 84010102, 13847801, 84422008, 19090525, 28870403, 86005480], [51885700, 34588771, 21295579, 96831132], [92012234, 99995213, 76712861], [86134882, 19646230, 27059426], [77233811, 11210891, 95162387, 37218200, 20174041, 69306754, 64292332, 56017531, 85476594], [18765083, 23713900], [10135556, 41875934, 12452731]]



lista_residentes = []
lista_id_residentes = []
lista_comodos_residentes =[]
lista_id_comodos = []

lista_franquia=[]
lista_id_franquia=[]
lista_child_franq=[]
lista_child_id_franq = []

lista_obj_id=[]
lista_obj=[]
lista_obj_pos=[]


lista_residentes1 = []
lista_id_residentes1 = []
lista_comodos_residentes1 =[]
lista_id_comodos1 = []

lista_franquia1=[]
lista_id_franquia1=[]
lista_child_franq1=[]
lista_child_id_franq1 = []

lista_obj_id1=[]
lista_obj1=[]
lista_obj_pos1=[]


try:
    with open("lista_franquia.txt", "r") as fp:
        lista_franquia = json.load(fp)
    with open("lista_id_franquia.txt", "r") as fp:
        lista_id_franquia = json.load(fp)       
    with open("lista_child_id_franq.txt", "r") as fp:
        lista_child_id_franq = json.load(fp)    
    with open("lista_child_franq.txt", "r") as fp:
        lista_child_franq = json.load(fp)     

    with open("lista_comodos_residentes.txt", "r") as fp:
        lista_comodos_residentes = json.load(fp)
    
    with open("lista_comodos_residentes_local.txt", "r") as fp:
        lista_comodos_residentes_Local = json.load(fp)

    with open("lista_id_comodos.txt", "r") as fp:
        lista_id_comodos = json.load(fp)
    with open("lista_residentes.txt", "r") as fp:
        lista_residentes = json.load(fp)
    with open("lista_id_residentes.txt", "r") as fp:
        lista_id_residentes = json.load(fp)

    with open("lista_obj.txt", "r") as fp:
        lista_obj = json.load(fp)
    with open("lista_obj_id.txt", "r") as fp:
        lista_obj_id = json.load(fp)
    with open("lista_obj_pos.txt", "r") as fp:
        lista_obj_pos = json.load(fp)            


except:
    print("Listas não existem")
    pass


# def timer():
#     while True:
#         # print("Criando Listas")
#         cria_listas()
#         time.sleep(15)   # 3 min.


##Configuracao do texto
# fontScale 
fontScale = 1
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
#Configuracao da exibicao dos pontos no opencv
LINE_THICKNESS = 1
LINE_COLOR = (255,255,255) #RGB

tempo = time.time()

avg_fps = deque(maxlen=2) #lista para calcular a media de fps
fps_time = 0
avg_fps_calc = 0

url, franquia, residente, comodo = dados_iniciais()
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    card1 = st.empty()
with col2:
    card2 = st.empty()
with col3:
    card3 = st.empty()

def obj_viewr(obj):
    print("gesge",obj)

# def opncv(frame_from_mqtt, iddd):





def opncv(frame_from_mqtt, iddd, obj,llm,people_position,people_status):
    global resolucao
    global IMG_BACKGROUND
    # print(iddd, obj)
    img = np.zeros((resolucao[1],resolucao[0],3), np.uint8) # imagem fundo preto

    list_toral=[]
    list_view_total=[]

    people_landmarks_list = []
    list_visibility_2=[]
    print("people_status",people_status)
    if len(people_position)>0:

        for ii in range(len(people_position)):
            
            # print("people_status2",people_status[ii][0])
            # if str(people_status[0]) == 'Em Pe':
            #     cv2.circle(img, (int(people_position[ii][0]/1920*resolucao[0]), int(people_position[ii][1]/1080*resolucao[1])),  25, (0, 255, 0), 10)
            # elif str(people_status[0]) == 'Sentado':
            #     cv2.circle(img, (int(people_position[ii][0]/1920*resolucao[0]), int(people_position[ii][1]/1080*resolucao[1])),  25, (255, 0, 0), 10)
            # elif str(people_status[0]) == 'Curvado':
            #     cv2.circle(img, (int(people_position[ii][0]/1920*resolucao[0]), int(people_position[ii][1]/1080*resolucao[1])),  25, (255, 255, 255), 10)    
            # else:
            #     cv2.circle(img, (int(people_position[ii][0]/1920*resolucao[0]), int(people_position[ii][1]/1080*resolucao[1])),  25, (0, 0, 255), 10)


            if str(people_status[0]) == 'Em Pe':
                cv2.circle(img, (int(people_position[ii][0]/1280*resolucao[0]), int(people_position[ii][1]/720*resolucao[1])),  25, (0, 255, 0), 10)
            elif str(people_status[0]) == 'Sentado':
                cv2.circle(img, (int(people_position[ii][0]/1280*resolucao[0]), int(people_position[ii][1]/720*resolucao[1])),  25, (255, 0, 0), 10)
            elif str(people_status[0]) == 'Curvado':
                cv2.circle(img, (int(people_position[ii][0]/1280*resolucao[0]), int(people_position[ii][1]/720*resolucao[1])),  25, (255, 255, 255), 10)    
            else:
                cv2.circle(img, (int(people_position[ii][0]/1280*resolucao[0]), int(people_position[ii][1]/720*resolucao[1])),  25, (0, 0, 255), 10)
            # print(people_position[ii][0], people_position[ii][1])    
            # print(people_position[ii][0], people_position[ii][1])


    if llm is not None:
        try:
            for many_boxes in range(len(llm['bounding_boxes_pessoas'])):
                    # print(llm['bounding_boxes'][many_boxes])
                boxes=llm['bounding_boxes_pessoas'][many_boxes]

                poses = llm['poses'][many_boxes]


                xx1 = int(boxes[1]/1000*resolucao[0])
                yx1 = int(boxes[0]/1000*resolucao[1])
                xx2 = int(boxes[3]/1000*resolucao[0])
                yx2 = int(boxes[2]/1000*resolucao[1])

         

                labbel2 = str(boxes[4])
                poses_label = str(poses)
                label_total = labbel2+'_'+poses_label
                cv2.rectangle(img,(xx1,yx1),(xx2,yx2),(200,255,0),4)
                cv2.putText(img, label_total, (xx1, yx1+30),cv2.FONT_HERSHEY_SIMPLEX, 1, (200,123,0), 2)  
        except:
            pass        
   


    if obj is not None:    
        for many_obj in range(len(obj)):
            dic_many_obj=list_obj[many_obj]
            # print(dic_many_obj)
            x1 = int(dic_many_obj['pos'][0]*resolucao[0])
            y1 = int(dic_many_obj['pos'][1]*resolucao[1])
            x2 = int(dic_many_obj['pos'][2]*resolucao[0])
            y2 = int(dic_many_obj['pos'][3]*resolucao[1])

            # x1 = int(dic_many_obj['left']*resolucao[0])
            # y1 = int(dic_many_obj['top']*resolucao[1])
            # x2 = int(dic_many_obj['right']*resolucao[0])
            # y2 = int(dic_many_obj['bottom']*resolucao[1])
            labbel = str(dic_many_obj['label'])
            # if 'label' in dic_many_obj:

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),4)
            cv2.putText(img, labbel, (x1, y1+30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)  
            #     print(dic_many_obj['label'])







    #print AREA
    for i in range(len(lista_id_comodos)):
            # print()
            for nn in range(len(lista_id_comodos[i])):
                # print(lista_id_comodos[i][nn])
                
                if (str(lista_id_comodos[i][nn])==str(iddd)):
                    for p in range(len(lista_obj[i][nn])):
                        xxyy=[]
                        for oi in range (len(lista_obj_pos[i][nn][p])):
                            # print(lista_obj_pos[i][nn][p][oi])
                            xx = int(lista_obj_pos[i][nn][p][oi][0]*resolucao[0])
                            yy = int(lista_obj_pos[i][nn][p][oi][1]*resolucao[1])
                            xxyy.append([xx,yy])
                        points=np.array(xxyy)
                        pts = points.reshape(-1,1,2)
                        cv2.polylines(img, np.int32([pts]), isClosed=True, color=(255,255,255), thickness = 2)
                        cv2.putText(img, lista_obj[i][nn][p], (int(lista_obj_pos[i][nn][p][0][0]*resolucao[0]), int(lista_obj_pos[i][nn][p][0][1]*resolucao[1])+30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)    
                    
    return img

ccomodo=0

while True:
    
    Lista3 = []
    # frames = []
    Lista4 = []

    statusJson = {}


################### DADOS MQTT ##################
    # if (flag_connected == 0):
    # # if (flag_connected == 0 and (time.time() - tempo) > 10):
    #     client.reconnect()
    # #     # ic('connecting to broker')
    # #     tempo = time.time()
    # else:
    # #     client.loop(0.05)
    #     client.loop_start()
    try:
        if(len(frames) > 0): #verifica se ha dados na lista
            try:
                data_mqtt = frames.pop()
                # print(frames.pop())
                data_mqtt = json.loads(data_mqtt)
                # print(data_mqtt)

                
                    
            except:
                print(traceback.format_exc())
                pass
################### DADOS API ##################        
            response = requests.get(url)
            data_api = response.json()
            res_data=""

            dados_residentes_franquia = [record.get("Residentes") for record in data_api if record.get("Franquia") == franquia]
            dados_residente = [record for record in dados_residentes_franquia[0] if record.get("Residente") == residente]
            for iid in range(len(dados_residente)):
                res_data=dados_residente[iid]['Resid_data']
                # print(dados_residente[iid]['Resid_data'])
            dado_comodo = [record for record in dados_residente[0]['comodos'] if record.get("local") == comodo]
            deviceee = dado_comodo[0]['deviceId']
            ccomodo = dado_comodo[0]

            # if "mediapipe" in data_mqtt:
            #     if (str(dado_comodo[0]['deviceId']) ==str(data_mqtt['device_id'])):

            #         print("data_mqtt",data_mqtt['mediapipe'])
                
###################MOSTRAR NO PAINEL##################
            with card1:
                with st.container():
                    st.write(' JSON Residente ')
                    st.write(res_data)
                    # print(data_mqtt)
                    # try:
                    #     if data_mqtt['device_id']==str(dado_comodo[0]['deviceId']):
                    #         st.write('Timestamp: ',int(float(data_mqtt['timestamp'])))
                    #         st.write(data_mqtt)
                    # except:
                    #     pass

                    # try:
                    #     if data_mqtt['device']==str(dado_comodo[0]['deviceId']):
                    #         st.write('Timestamp: ',int(float(data_mqtt['timestamp'])))
                    #         st.write(data_mqtt)
                    #         # print(data_mqtt)
                    #         # st.write("Achou", count)
                    # except:
                    #     pass


            with card2:
                with st.container():
                    st.write('**JSON Cômodo**')
                    st.write('Timestamp: ',int([record.get("timestamp") for record in data_api][0]))
                    st.write(dado_comodo[0])
                    # print(data_mqtt)

            
            with card3:
                with st.container():
                    st.write('**Visualização Cômodo**')
                    objj = dado_comodo[0]
                    people_status=[]
                    if 'poses'in objj.keys():
                        # print(objj['poses'])
                        people_status=objj['poses']

                    pose=[]
                    if 'people_position'in objj.keys():
                        people_position=objj['people_position']
                    else:
                        people_position=[]

                        # print(people_position)

                    if 'llm'in objj.keys():
                        llm=objj['llm']
                        # print(llm)
                    # if "mediapipe" in data_mqtt:
                    #     print("MEDIAPIPE",data_mqtt["mediapipe"])
                    if 'obj' in objj.keys():
                        list_obj=objj['obj']
                        # print(list_obj)
                        
                        if type(list_obj) is list:   
                            if "mediapipe" in data_mqtt:
                                pose=[]
                                # if ("mediapipe" in data_mqtt):
                                for keytt, valutt in data_mqtt.items():
                                    if valutt==str(deviceee):
                                        media=data_mqtt['mediapipe']
                                        # print("ccomodo",ccomodo)
                                        if len(media)>0:
                                            pose=media
                                        else:
                                            pose=[] 

                                            # print(deviceee,(data_mqtt))                            
                            imgs = opncv(pose,deviceee, list_obj, llm,people_position,people_status)
                            # imgs = opncv(pose,deviceee, list_obj)
                           
                            st.image(imgs)

                            # for many_obj in range(len(list_obj)):
                            #     dic_many_obj=list_obj[many_obj]
                            #     if 'label' in dic_many_obj:
                            #         print(dic_many_obj['label'])


                    # try:
                    #     if data_mqtt['device_id']==str(dado_comodo[0]['deviceId']):
                           
                    #         for key, value in data_mqtt.items():
                    #             # print(key)
                    #             if (key=="device_id"):
                    #                 # print(value)
                    #                 iddd = value
                    #             if (key=="obj"):
                    #                 # imgs2 = obj_viewr(value )
                    #                 # print(value)
                    #                 imgs2 = []
                                   

                    #             if (key=="mediapipe"):
                    #                 # print(iddd, dado_comodo[0])
                                    
                    #                 # if "obj" in objj:
                    #                 #     print(objj)

                    #                 imgs = opncv(value,iddd, imgs2)
                    #                 # imgs = opncv(value,iddd)
                    #                 st.image(imgs)
                    # except:
                    #     pass                    


        Lista = []
        Lista3 = []
######################
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        pass

    time.sleep(0.15)

