# Import Libraries
import cv2
import face_recognition
import matplotlib.pyplot as plt
import os ,numpy ,argparse
from datetime import datetime

# Take input Video from terminal(Option)
parser = argparse.ArgumentParser()

parser.add_argument('-i','--input',default="./Indian_Team1.mp4",type=str,help="Path to Imput video File(Option)")
arg = vars(parser.parse_args())

# ------- Pre-Process Operations on Input Images  -------

known_img_path =os.path.join( os.getcwd(), 'images')
# print(os.listdir(known_img_path))

 #--------- Some information about our Candidates in Image --------
Player_Info = { 'A Kumble':['india','Coach'] ,'Abdul kalam':['india','Scinetist'] ,'B Kumar':['india','Bowler'] ,'Dhawan':['india','Batsman'] ,'Dhoni':['india','Captain'] ,'Elon':['USA','Inventor'] ,'Hardik Panya':['india','All-Rounder'] ,'Jusprit Bumra':['india','Bowler'] ,'KL Rahul':['india','batsman'] ,'M Shami':['india','Bowler'] ,'R Ashwin':['india','Coach'] ,'R Jadeja':['India','All-Rounder'] ,'Rishabh Pant':['india','Batsman'] ,'Rohit Sharma':['india','Batsman'] ,'Virat':['india','Vice-Captain']  ,'U Yadav':['india','Bowler'] , }
# print(Player_Info)

def Mark_present(Name): 
    with open(file_name,'r+') as f:
        All_Names= []
        for pair in f.readlines():
            All_Names.append( pair.split(',')[0] )
        if Name not in All_Names:
            now = datetime.now()
            now = now.strftime("%H:%M:%S")
            Country = Player_Info[Name][0]
            Speciality = Player_Info[Name][1]
            f.writelines(f'\n{Name},{Country},{Speciality},{now}')        

 # Make the Class of their name 
All_images = os.listdir(known_img_path)
Classes=[]
for i in All_images:
    Classes.append(i.split('.')[0])
# print(Classes) 

# ------------ Find Encodings of Images ------------
known_faces_enco=[]

for i in All_images:
    img = cv2.imread(os.path.join(known_img_path , i))
    img_encoding = face_recognition.face_encodings(img)
    known_faces_enco.append(img_encoding[0])


# ------------check if .csv file is present if not then Create One
file_name ='MyAttandance.csv'
if file_name not in os.listdir(os.getcwd()):
    open(file_name,'x')
    with open(file_name,'r+') as f:
        Name="Name"
        Country='Country'
        Speciality='Speciality'
        Time="Time"
        f.writelines(f'{Name},{Country},{Speciality},{Time}')

# ------------- Take Input to Process  -----------
input_ = arg['input']
video  =  cv2.VideoCapture(input_ if(input_) else 0)

while(1):
    if(cv2.waitKey(1)==ord('q')):
        break
        
    is_cap,frame= video.read()
#     if(is_cap):
#         print("video capturing...")
            
    face_loc = face_recognition.face_locations(frame) # locations of All faces in the fra
    face_enco = face_recognition.face_encodings(frame,face_loc)  # Encodnings of All faces in the fra
    result = []
    for encode_face, loc_face in zip(face_enco,face_loc):
        
        
        maches = face_recognition.compare_faces(known_faces_enco, encode_face,tolerance=0.5)
        Dis = face_recognition.face_distance( known_faces_enco, encode_face )
        Person_idx = numpy.argmin(Dis)
        if( maches[ Person_idx ] ): #if Person Detected
            
            y1,x2,y2,x1 = loc_face
            cv2.rectangle(frame ,(x1,y1),(x2,y2),(0,255,0),2 )
            cv2.rectangle(frame ,(x1,y2-15),(x2,y2),(0,255,0),cv2.FILLED )
            text = f"{Classes[Person_idx]}"
            cv2.putText(frame,text, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (200,0,0),1)
            Mark_present( Classes[Person_idx] ) # mark Present if this is new
#         print(f"maches {maches}")
#         print(f"Dis {Dis}")
    

    
        
    cv2.imshow("video",frame)    
        
        
video.release()
cv2.destroyAllWindows()


# See All saved People in the .csv file Recognised by their Faces
Entry_names= []
with open(file_name,'r+') as f:
        Entry_names.append([ i for i in f.read().splitlines()] )

print(Entry_names)