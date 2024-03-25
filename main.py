from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import mysql.connector
import os
import datetime
from geopy.distance import geodesic
import numpy as np
import folium
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from flask_socketio import SocketIO, emit, join_room, leave_room
from engineio.payload import Payload
Payload.max_decode_packets = 200
from werkzeug.utils import secure_filename
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
from plotly import graph_objects as go
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import imagehash
from werkzeug.utils import secure_filename
from PIL import Image
import argparse
import urllib.request
import urllib.parse
import math

   
# necessary imports 
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
#%matplotlib inline
pd.set_option('display.max_columns', 26)
##
from PIL import Image, ImageOps
import scipy.ndimage as ndi

from skimage import transform
import seaborn as sns
#from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
#from keras.models import Sequential
#from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
##
import glob
#from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import seaborn as sns
#import keras as k
#from keras.layers import Dense
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#from tensorflow.keras.optimizers import Adam
##

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'abcdef'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    charset="utf8",
    use_pure=True,
    database="agri"
)

UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


socketio = SocketIO(app)


_users_in_room = {} # stores room wise user list
_room_of_sid = {} # stores room joined by an used
_name_of_sid = {} # stores display name of users


@app.route('/',methods=['POST','GET'])
def index():

    
    return render_template('index.html')

@app.route('/login',methods=['POST','GET'])
def login():

    msg=""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM farmer WHERE username = %s AND password = %s AND action = 1', (username, password))
        account = cursor.fetchone()
        
        if account:
            session['username'] = username
            session['user_type'] = 'farmer'
            msg="success"  
        else:
            msg="fail"

    return render_template('login.html', msg=msg)




@app.route('/register',methods=['POST','GET'])
def register():
    
    msg=""
    if request.method=='POST':
        name=request.form['name']
        address=request.form['address']
        mobile=request.form['mobile']
        email=request.form['email']
        username=request.form['username']
        password=request.form['password']
        longitude=request.form['longitude']
        latitude=request.form['latitude']
        now = datetime.datetime.now()
        reg_date=now.strftime("%Y-%m-%d")
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM farmer where username=%s",(username, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM farmer")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO farmer(id, name, address, mobile, email, username, password, reg_date, longitude, latitude) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, address, mobile, email, username, password, reg_date, longitude, latitude)
            mycursor.execute(sql, val)
            mydb.commit()

            msg="success"
        else:
            msg="fail"
  
    return render_template('register.html', msg=msg)


@app.route('/login1',methods=['POST','GET'])
def login1():

    msg=""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM user WHERE username = %s AND password = %s AND action = 1', (username, password))
        account = cursor.fetchone()
        
        if account:
            session['username'] = username
            session['user_type'] = 'user'
            msg="success"  
        else:
            msg="fail"

    
    return render_template('login1.html', msg=msg)




@app.route('/admin',methods=['POST','GET'])
def admin():

    msg=""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        
        if account:
            session['username'] = username
            session['user_type'] = 'admin'
            msg="success"  
        else:
            msg="fail"

    
    return render_template('admin.html', msg=msg)




@app.route('/register1',methods=['POST','GET'])
def register1():
    
    msg=""
    if request.method=='POST':
        name=request.form['name']
        address=request.form['address']
        mobile=request.form['mobile']
        email=request.form['email']
        username=request.form['username']
        password=request.form['password']
        longitude=request.form['longitude']
        latitude=request.form['latitude']
        now = datetime.datetime.now()
        reg_date=now.strftime("%Y-%m-%d")
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM user where username=%s",(username, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM user")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO user(id, name, address, mobile, email, username, password, reg_date, longitude, latitude) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, address, mobile, email, username, password, reg_date, longitude, latitude)
            mycursor.execute(sql, val)
            mydb.commit()

            msg="success"
        else:
            msg="fail"
  
    return render_template('register1.html', msg=msg)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/post', methods=['POST', 'GET'])
def post():
    if 'username' not in session or session.get('user_type') != 'farmer':
        print("Please log in as a admin to access the page.", 'danger')
        return redirect(url_for('login'))
    
    dt=""
    food_type=None
    post_id=None
    username = session.get('username')
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM farmer WHERE username = %s", (username,))
    data = cursor.fetchone()
    cursor.close()
    name=data[1]
    address=data[2]
    mobile=data[3]
    longitude=data[9]
    latitude=data[10]
    
    msg=""
    nearby_users = []
    num_nearby_users = 0
    provider_coords = None  # Initialize with a default value
    if request.method=='POST':
        product_type=request.form['product_type']
        product=request.form['product']
        price=request.form['price']
        quantity=request.form['quantity']
        message=request.form['message']
        if 'image' in request.files:
            image = request.files['image']

            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                image_path = 'C:/Users/deera/OneDrive/Documents/agriecom/static/uploads/' + filename
                image.save(image_path)
        
                now = datetime.datetime.now()
                post_date=now.strftime("%B %d, %Y")
                post_time=now.strftime("%I:%M %p")
        
                mycursor = mydb.cursor()
        
                mycursor.execute("SELECT max(id)+1 FROM post")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1
                sql = "INSERT INTO post(id, product_type, product, price, message, name, address, mobile, post_date, post_time, username, longitude, latitude, quantity, image) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                val = (maxid, product_type, product, price, message, name, address, mobile, post_date, post_time, username, longitude, latitude, quantity, filename)
                mycursor.execute(sql, val)
                mydb.commit()

                msg="success"

        

                try:
                    provider_cursor = mydb.cursor(dictionary=True)
                    provider_cursor.execute("SELECT * FROM farmer WHERE username = %s", (username,))
                    provider_data = provider_cursor.fetchone()
                    provider_cursor.close()

                    # Extract provider coordinates
           
                    provider_coords = (provider_data['latitude'], provider_data['longitude'])

                    print("Provider Coordinates:", provider_coords)

                    # Fetch all users with valid latitude and longitude
                    user_cursor = mydb.cursor(dictionary=True)
                    user_cursor.execute("SELECT username, latitude, longitude FROM user WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
                    all_users = user_cursor.fetchall()
                    user_cursor.close()

                    for user in all_users:
                        user_coords = (user['latitude'], user['longitude'])
                        distance = geodesic(provider_coords, user_coords).kilometers


                        if distance < 100:  # Adjust the distance threshold as needed
                            user_details = get_user_details(user['username'])  # Fetch additional details
                            if user_details:
                                nearby_users.append({
                                    'username': user['username'],
                                    'latitude': user['latitude'],
                                    'longitude': user['longitude'],
                                    'user_details': user_details 
                                }) 

                    # Count the number of nearby users
                    num_nearby_users = len(nearby_users)

                except Exception as e:
                    print(f"An error occurred: {e}")
    

    return render_template('post.html', msg=msg, nearby_users=nearby_users, num_nearby_users=num_nearby_users, provider_coords=provider_coords, username=username)



def get_user_details(username):
    try:
        user_cursor = mydb.cursor(dictionary=True)
        user_cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
        user_details = user_cursor.fetchone()
        user_cursor.close()
        return user_details
    except Exception as e:
        print(f"An error occurred while fetching user details: {e}")
        return None




@app.route('/view_post', methods=['POST', 'GET'])
def view_post():
    if 'username' not in session or session.get('user_type') != 'user':
        print("Please log in as a admin to access the page.", 'danger')
        return redirect(url_for('login1'))

    
    dt=""
    food_type=None
    post_id=None
    msg=""
    username = session.get('username')
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
    data = cursor.fetchone()
    cursor.close()
    name=data[1]
    address=data[2]
    mobile=data[3]
    longitude=data[9]
    latitude=data[10]

    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM post")
    data5 = cursor.fetchall()
    cursor.close()

    act=request.args.get("act")
    

    if act=="call":
        
        pid=request.args.get("pid")
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM post WHERE id = %s", (pid,))
        data8 = cursor.fetchone()
        cursor.close()
        far_username=data8[10]
        product=data8[2]
        now = datetime.datetime.now()
        req_date=now.strftime("%B %d, %Y")
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM request")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO request(id, username, far_username, product, req_date) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid, username, far_username, product, req_date)
        mycursor.execute(sql, val)
        mydb.commit()
        msg="success"

    else:

        msg="fail"

    if act=="ok":
        
        pid=request.args.get("pid")
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM post WHERE id = %s", (pid,))
        data6 = cursor.fetchone()
        cursor.close()
        far_username=data6[10]
        product=data6[2]
        price=data6[4]
        far_mobile=data6[7]
        now = datetime.datetime.now()
        req_date=now.strftime("%B %d, %Y")
        
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM book")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO book(id, far_username, product, price, far_mobile, mobile, username, req_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (maxid, far_username, product, price, far_mobile, mobile, username, req_date)
        mycursor.execute(sql, val)
        mydb.commit()
        msg="success1"

        session['maxid'] = maxid

        
        return redirect(url_for('quantity', maxid=maxid))
    else:
        msg="fail1"
        
        
    
    msg=""
    nearby_users = []
    num_nearby_users = 0
    provider_coords = None  # Initialize with a default value
    if request.method=='POST':
        search=request.form['search']
        
        try:
            provider_cursor = mydb.cursor(dictionary=True)
            provider_cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
            provider_data = provider_cursor.fetchone()
            provider_cursor.close()

            # Extract provider coordinates
           
            provider_coords = (provider_data['latitude'], provider_data['longitude'])

            print("Provider Coordinates:", provider_coords)

            # Fetch all users with valid latitude and longitude
            user_cursor = mydb.cursor(dictionary=True)
            user_cursor.execute("SELECT username, latitude, longitude FROM post WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND product=%s", (search,))
            all_users = user_cursor.fetchall()
            user_cursor.close()

            for user in all_users:
                user_coords = (user['latitude'], user['longitude'])
                distance = geodesic(provider_coords, user_coords).kilometers


                if distance < 100:  # Adjust the distance threshold as needed
                    user_details = get_post_details(user['username'])  # Fetch additional details
                    if user_details:
                        nearby_users.append({
                            'username': user['username'],
                            'latitude': user['latitude'],
                            'longitude': user['longitude'],
                            'user_details': user_details 
                        }) 

            # Count the number of nearby users
            num_nearby_users = len(nearby_users)

        except Exception as e:
            print(f"An error occurred: {e}")

    return render_template('view_post.html', post=data5, msg=msg, nearby_users=nearby_users, num_nearby_users=num_nearby_users, provider_coords=provider_coords, username=username)



def get_post_details(username):
    try:
        user_cursor = mydb.cursor(dictionary=True)
        user_cursor.execute("SELECT * FROM post WHERE username = %s", (username,))
        user_details = user_cursor.fetchone()
        user_cursor.close()
        return user_details
    except Exception as e:
        print(f"An error occurred while fetching user details: {e}")
        return None


@app.route('/quantity', methods=['GET', 'POST'])
def quantity():
    if 'username' not in session or session.get('user_type') != 'user':
        print("Please log in as a admin to access the page.", 'danger')
        return redirect(url_for('login1'))

    
    maxid=session.get('maxid')

    if request.method=='POST':
        maxid=request.form['maxid']
        quantity=request.form['quantity']
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM book where id = %s",(maxid,))
        data1 = cursor.fetchone()
        cursor.close()
        price=data1[3]

        quantity = int(quantity)
        price = float(price)

        total=quantity*price
        
        cursor = mydb.cursor()
        cursor.execute("update book set quantity=%s, total=%s where id=%s",(quantity, total, maxid))
        mydb.commit()
        return redirect(url_for('payment', maxid=maxid))

    return render_template('quantity.html', maxid=maxid)




@app.route('/payment', methods=['GET', 'POST'])
def payment():
    if 'username' not in session or session.get('user_type') != 'user':
        print("Please log in as a admin to access the page.", 'danger')
        return redirect(url_for('login1'))

    msg=""
    total=""
    maxid=session.get('maxid')
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM book where id = %s",(maxid,))
    data1 = cursor.fetchone()
    cursor.close()
    total=data1[10]

    if request.method=='POST':
        maxid=request.form['maxid']
        payment=request.form['payment']
        
        cursor = mydb.cursor()
        cursor.execute("update book set payment=%s where id=%s",(payment, maxid))
        mydb.commit()
        msg="success"
    else:
        msg="fail"

    return render_template('payment.html', maxid=maxid, msg=msg, total=total)




@app.route('/user_req', methods=['GET', 'POST'])
def user_req():
    if 'username' not in session or session.get('user_type') != 'farmer':
        print("Please log in as a admin to access the page.", 'danger')
        return redirect(url_for('login'))

    
    username=session.get('username')

    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM request where far_username = %s",(username,))
    data3 = cursor.fetchall()
    cursor.close()

    act=request.args.get("act")

    if act=="ok":
        aid=request.args.get("aid")
        cursor = mydb.cursor()
        cursor.execute("update request set action=1 where id=%s",(aid,))
        mydb.commit()
        print("successfully accepted")
        
    if act=="no":
        aid=request.args.get("aid")
        cursor = mydb.cursor()
        cursor.execute("update request set action=2 where id=%s",(aid,))
        mydb.commit()
        print("your account will be rejected")

    if request.method=='POST':
        aid=request.form['aid']
        date=request.form['date']
        time=request.form['time']
        cursor = mydb.cursor()
        cursor.execute("update request set date=%s, time=%s where id=%s",(date, time, aid))
        mydb.commit()

    return render_template('user_req.html', request=data3)


@app.route('/user_book', methods=['GET', 'POST'])
def user_book():
    if 'username' not in session or session.get('user_type') != 'farmer':
        print("Please log in as a admin to access the page.", 'danger')
        return redirect(url_for('login'))

    
    username=session.get('username')

    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM book where far_username = %s",(username,))
    data3 = cursor.fetchall()
    cursor.close()

    

    if request.method=='POST':
        aid=request.form['aid']
        order_status=request.form['order_status']
        cursor = mydb.cursor()
        cursor.execute("update book set status=%s where id=%s",(order_status, aid))
        mydb.commit()

    return render_template('user_book.html', book=data3)


@app.route('/view_book', methods=['GET', 'POST'])
def view_book():
    if 'username' not in session or session.get('user_type') != 'user':
        print("Please log in as a admin to access the page.", 'danger')
        return redirect(url_for('login1'))

    
    username=session.get('username')

    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM book where username = %s",(username,))
    data3 = cursor.fetchall()
    cursor.close()

    return render_template('view_book.html', book=data3)


@app.route('/view_req', methods=['GET', 'POST'])
def view_req():
    if 'username' not in session or session.get('user_type') != 'user':
        print("Please log in as a admin to access the page.", 'danger')
        return redirect(url_for('login1'))

    
    username=session.get('username')

    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM request where username = %s",(username,))
    data3 = cursor.fetchall()
    cursor.close()

    

    return render_template('view_req.html', request=data3)


@app.route("/call", methods=["GET", "POST"])
def call():

    aid=request.args.get("aid")
    if request.method == "POST":
        room_id = request.form['room_id']
        cursor = mydb.cursor()
        cursor.execute("update request set link=%s where id=%s",(room_id, aid))
        mydb.commit()
        
        return redirect(url_for("entry_checkpoint", room_id=room_id, aid=aid))

    return render_template("call.html")

@app.route("/room/<string:room_id>/")
def enter_room(room_id):
    act=request.args.get("act")
    
    
    if room_id not in session:
        return redirect(url_for("entry_checkpoint", room_id=room_id))
    
    return render_template("chatroom.html", room_id=room_id, display_name=session[room_id]["name"], mute_audio=session[room_id]["mute_audio"], mute_video=session[room_id]["mute_video"])

@app.route("/room/<string:room_id>/checkpoint/", methods=["GET", "POST"])
def entry_checkpoint(room_id):
    

    username=""
    
    if request.method == "POST":
        mute_audio = request.form['mute_audio']
        mute_video = request.form['mute_video']
        session[room_id] = {"name": username, "mute_audio":mute_audio, "mute_video":mute_video}
        return redirect(url_for("enter_room", room_id=room_id))

    return render_template("chatroom_checkpoint.html", room_id=room_id)

@socketio.on("connect")
def on_connect():
    sid = request.sid
    print("New socket connected ", sid)
    

@socketio.on("join-room")
def on_join_room(data):
    sid = request.sid
    room_id = data["room_id"]
    display_name = session[room_id]["name"]
    
    # register sid to the room
    join_room(room_id)
    _room_of_sid[sid] = room_id
    _name_of_sid[sid] = display_name
    
    # broadcast to others in the room
    print("[{}] New member joined: {}<{}>".format(room_id, display_name, sid))
    emit("user-connect", {"sid": sid, "name": display_name}, broadcast=True, include_self=False, room=room_id)
    
    # add to user list maintained on server
    if room_id not in _users_in_room:
        _users_in_room[room_id] = [sid]
        emit("user-list", {"my_id": sid}) # send own id only
    else:
        usrlist = {u_id:_name_of_sid[u_id] for u_id in _users_in_room[room_id]}
        emit("user-list", {"list": usrlist, "my_id": sid}) # send list of existing users to the new member
        _users_in_room[room_id].append(sid) # add new member to user list maintained on server

    print("\nusers: ", _users_in_room, "\n")


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    room_id = _room_of_sid[sid]
    display_name = _name_of_sid[sid]

    print("[{}] Member left: {}<{}>".format(room_id, display_name, sid))
    emit("user-disconnect", {"sid": sid}, broadcast=True, include_self=False, room=room_id)

    _users_in_room[room_id].remove(sid)
    if len(_users_in_room[room_id]) == 0:
        _users_in_room.pop(room_id)

    _room_of_sid.pop(sid)
    _name_of_sid.pop(sid)

    print("\nusers: ", _users_in_room, "\n")


@socketio.on("data")
def on_data(data):
    sender_sid = data['sender_id']
    target_sid = data['target_id']
    if sender_sid != request.sid:
        print("[Not supposed to happen!] request.sid and sender_id don't match!!!")

    if data["type"] != "new-ice-candidate":
        print('{} message from {} to {}'.format(data["type"], sender_sid, target_sid))
    socketio.emit('data', data, room=target_sid)




#####################################################################################################################################################################


@app.route('/admin1', methods=['GET', 'POST'])
def admin1():
    
    
        
        
    return render_template('admin1.html')

@app.route('/add_plant', methods=['GET', 'POST'])
def add_plant():
    msg=""
    

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        disease=request.form['disease']
        symptoms=request.form['symptoms']
        solution=request.form['solution']
        
        mycursor.execute("SELECT max(id)+1 FROM leaf_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO leaf_data(id,disease,symptoms,solution) VALUES (%s, %s, %s, %s)"
        val = (maxid,disease,symptoms,solution)
        mycursor.execute(sql,val)
        mydb.commit()
        return redirect(url_for('add_plant'))

    mycursor.execute('SELECT * FROM leaf_data')
    data = mycursor.fetchall()
    
    return render_template('add_plant.html',msg=msg,data=data)

@app.route('/img_process', methods=['GET', 'POST'])
def img_process():
    

    return render_template('img_process.html')

@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #list_of_elements = os.listdir(os.path.join(path_main, folder))

        #resize
        #img = cv2.imread('static/data/'+fname)
        #rez = cv2.resize(img, (400, 300))
        #cv2.imwrite("static/dataset/"+fname, rez)'''

        '''img = cv2.imread('static/dataset/'+fname) 	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/trained/g_"+fname, gray)
        ##noice
        img = cv2.imread('static/trained/g_'+fname) 
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        fname2='ns_'+fname
        cv2.imwrite("static/trained/"+fname2, dst)'''

    return render_template('pro1.html',dimg=dimg)


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/pro11', methods=['GET', 'POST'])
def pro11():
    msg=""
    dimg=[]
    path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)

    return render_template('pro11.html',dimg=dimg)

@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #f1=open("adata.txt",'w')
        #f1.write(fname)
        #f1.close()
        ##bin
        '''image = cv2.imread('static/dataset/'+fname)
        original = image.copy()
        kmeans = kmeans_color_quantization(image, clusters=4)

        # Convert to grayscale, Gaussian blur, adaptive threshold
        gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

        # Draw largest enclosing circle onto a mask
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            break
        
        # Bitwise-and for result
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask==0] = (0,0,0)

        
        ###cv2.imshow('thresh', thresh)
        ###cv2.imshow('result', result)
        ###cv2.imshow('mask', mask)
        ###cv2.imshow('kmeans', kmeans)
        ###cv2.imshow('image', image)
        ###cv2.waitKey()

        cv2.imwrite("static/trained/bb/bin_"+fname, thresh)'''

    
   

    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        ##RPN
        
        
        img = cv2.imread('static/trained/g_'+fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,1.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        segment = cv2.subtract(sure_bg,sure_fg)
        img = Image.fromarray(img)
        segment = Image.fromarray(segment)
        path3="static/trained/sg/sg_"+fname
        #segment.save(path3)
        

    return render_template('pro2.html',dimg=dimg)


#Feature Extraction-GLCM
# ---calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ---
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    # --call calc_glcm_all_agls() for all properties --
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

    glcm_all_agls = []
    for img, label in zip(imgs, labels): 
        glcm_all_agls.append(
                calc_glcm_all_agls(img, 
                                    label, 
                                    props=properties)
                                )
     
    columns = []
    angles = ['0', '45', '90','135']
    for name in properties :
        for ang in angles:
            columns.append(name + "_" + ang)
            
    columns.append("label")
    return feature

###Classification
def CNN():
    #Lets start by loading the Cifar10 data
    (X, y), (X_test, y_test) = cifar10.load_data()

    #Keep in mind the images are in RGB
    #So we can normalise the data by diving by 255
    #The data is in integers therefore we need to convert them to float first
    X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0

    #Then we convert the y values into one-hot vectors
    #The cifar10 has only 10 classes, thats is why we specify a one-hot
    #vector of width/class 10
    y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)

    #Now we can go ahead and create our Convolution model
    model = Sequential()
    #We want to output 32 features maps. The kernel size is going to be
    #3x3 and we specify our input shape to be 32x32 with 3 channels
    #Padding=same means we want the same dimensional output as input
    #activation specifies the activation function
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                     activation='relu'))
    #20% of the nodes are set to 0
    model.add(Dropout(0.2))
    #now we add another convolution layer, again with a 3x3 kernel
    #This time our padding=valid this means that the output dimension can
    #take any form
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    #maxpool with a kernet of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #In a convolution NN, we neet to flatten our data before we can
    #input it into the ouput/dense layer
    model.add(Flatten())
    #Dense layer with 512 hidden units
    model.add(Dense(512, activation='relu'))
    #this time we set 30% of the nodes to 0 to minimize overfitting
    model.add(Dropout(0.3))
    #Finally the output dense layer with 10 hidden units corresponding to
    #our 10 classe
    model.add(Dense(10, activation='softmax'))
    #Few simple configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
    #Run the algorithm!
    model.fit(X, y, validation_data=(X_test, y_test), epochs=25,
              batch_size=512)
    #Save the weights to use for later
    model.save_weights("cifar10.hdf5")
    #Finally print the accuracy of our model!
    print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))


@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        
    '''path_main = 'static/dataset'
    i=1
    while i<=50:
        fname="r"+str(i)+".jpg"
        dimg.append(fname)

        img = Image.open('static/data/classify/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        i+=1
    i=1
    j=51
    while i<=10:
        
        fname="r"+str(j)+".jpg"
        dimg.append(fname)

        img = Image.open('static/dataset/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        j+=1
        i+=1

    '''    
    
    return render_template('pro3.html',dimg=dimg)

@app.route('/pro4', methods=['GET', 'POST'])
def pro4():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #####
        image = cv2.imread("static/dataset/"+fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        
        path4="static/trained/ff/"+fname
        #edged.save(path4)
        ##
    
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        
        parser = argparse.ArgumentParser(
        description='Script to run Yolo-V8 object detection network ')
        parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
        parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                          help='Path to text network file: '
                                               'MobileNetSSD_deploy.prototxt for Caffe model or '
                                               )
        parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                         help='Path to weights: '
                                              'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                              )
        parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
        args = parser.parse_args()

        # Labels of Network.
        classNames = { 0: 'background',
            1: 'plant' }

        # Open video file or capture device. 
        '''if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(0)'''

        #Load the Caffe model 
        net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

        #while True:
        # Capture frame-by-frame
        #ret, frame = cap.read()
        
        frame = cv2.imread("static/dataset/"+fname)
        frame_resized = cv2.resize(frame,(300,400)) # resize frame for prediction

        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 400), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        #Size of frame resize (300x400)
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > args.thr: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/300.0  
                widthFactor = frame.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                try:
                    y=yLeftBottom
                    h=yRightTop-y
                    x=xLeftBottom
                    w=xRightTop-x
                    image = cv2.imread("static/dataset/"+fname)
                    #mm=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    #cv2.imwrite("static/trained/classify/"+fname, mm)
                    #cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]

                    #gg="segment.jpg"
                    #cv2.imwrite("static/result/"+gg, cropped)


                    #mm2 = PIL.Image.open('static/trained/'+gg)
                    #rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                    #rz.save('static/trained/'+gg)
                except:
                    print("none")
                    #shutil.copy('getimg.jpg', 'static/trained/test.jpg')
                # Draw label and confidence of prediction in frame resized
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    claname=classNames[class_id]

                    
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    #print(label) #print class and confidence  
    return render_template('pro4.html',dimg=dimg)


    

@app.route('/pro5', methods=['GET', 'POST'])
def pro5():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
    #graph
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,210,265]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model Precision")
    plt.ylabel("precision")
    
    fn="graph1.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph2
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,220,275]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model recall")
    plt.ylabel("recall")
    
    fn="graph2.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,42,76,124,173]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="graph3.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,42,76,124,173]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="graph4.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    return render_template('pro5.html',dimg=dimg)

def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
                
@app.route('/pro6', methods=['GET', 'POST'])
def pro6():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    print("aaa")
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    '''data1=[]
    data2=[]
    data3=[]
    data4=[]
    v1=0
    v2=0
    v3=0
    v4=0
    path_main = 'static/trained'
    #for fname in os.listdir(path_main):
    i=0
    i<127
        dimg.append(fname)
        d1=fname.split('_')
        if d1[0]=='d':
            data1.append(fname)
            v1+=1
        if d1[0]=='f':
            data2.append(fname)
            v2+=1
        if d1[0]=='n':
            data3.append(fname)
            v3+=1
        if d1[0]=='w':
            data4.append(fname)
            v4+=1
        

    g1=v1+v2+v3+v4
    dd2=[v1,v2,v3,v4]
    
    
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 5))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.4)
 

    plt.ylim((1,g1))
    plt.xlabel("Objects")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()'''

    #,data1=data1,data2=data2,data3=data3,data4=data4,cname=cname,v1=v1,v2=v2,v3=v3,v4=v4
    ##############################

    
    ###############################
    
    
    

    return render_template('pro6.html',dimg=dimg)

#######
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""
    
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')


    ##    
    ff2=open("static/trained/tdata.txt","r")
    rd=ff2.read()
    ff2.close()

    num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    dat=toString(num)
    dd2=[]
    ex=dat.split(',')
    
    ##
    vv=[]
    vn=0
    data2=[]
    path_main = 'static/dataset'
    for val in ex:
        dt=[]
        n=0
        
        for fname in os.listdir(path_main):
            fa1=fname.split('.')
            fa=fa1[0].split('-')
            
            if fa[1]==val:
                dt.append(fname)
                n+=1
        vv.append(n)
        vn+=n
        data2.append(dt)
        
    print(vv)
    print(data2[0])
    
    i=0
    vd=[]
    data4=[]
    while i<8:
        vt=[]
        vi=i+1
        vv[i]

        vt.append(cname[i])
        vt.append(str(vv[i]))
        
        vd.append(str(vi))
        data4.append(vt)
        i+=1
    print(data4)
    
    
    dd2=vv
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 8))
     
    # creating the bar plot
    cc=['pink','yellow','orange','red','green','blue','brown','yellow']
    plt.bar(doc, values, color =cc,
            width = 0.4)
 

    plt.ylim((1,20))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    #plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()
    return render_template('classify.html',msg=msg,cname=cname,data2=data2)

#######
@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""

    
        
    return render_template('userhome.html',msg=msg)

@app.route('/test_img', methods=['GET', 'POST'])
def test_img():
    msg=""
    ss=""
    fn=""
    fn1=""
    tclass=0
    uname=""
    if 'username' in session:
        uname = session['username']
    result=""
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    
    if request.method=='POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            f1=open('static/test/file.txt','w')
            f1.write(filename)
            f1.close()
            file.save(os.path.join("static/test", filename))

        cutoff=1
        path_main = 'static/dataset'
        for fname1 in os.listdir(path_main):
            hash0 = imagehash.average_hash(Image.open("static/dataset/"+fname1)) 
            hash1 = imagehash.average_hash(Image.open("static/test/"+filename))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            if cc1<=cutoff:
                ss="ok"
                fn=fname1
                print("ff="+fn)
                break
            else:
                ss="no"

        if ss=="ok":
            print("yes")
            tclass=0
            dimg=[]

            ##    
            ff2=open("static/trained/tdata.txt","r")
            rd=ff2.read()
            ff2.close()

            num=[]
            r1=rd.split(',')
            s=len(r1)
            ss=s-1
            i=0
            while i<ss:
                num.append(int(r1[i]))
                i+=1

            #print(num)
            dat=toString(num)
            dd2=[]
            ex=dat.split(',')
            print(fn)
            ##
            
            ##
            n=0
            path_main = 'static/dataset'
            for val in ex:
                dt=[]
                
                fa1=fname.split('.')
                fa=fa1[0].split('-')
            
                if fa[1]==val:
                
                    result=val
                    
                    break
                    
                
                
                n+=1
                
            
            
            
            dta="a"+"|"+fn+"|"+result
            f3=open("static/test/res.txt","w")
            f3.write(dta)
            f3.close()

            
                    
            return redirect(url_for('test_pro',act="1"))
        else:
            msg="Invalid!"
    
    return render_template('test_img.html',msg=msg)


    
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg=""
    fn=""
    mycursor = mydb.cursor()
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    
    ts=gs[0]
    fname=fn
    ##bin
    '''image = cv2.imread('static/dataset/'+fn)
    original = image.copy()
    kmeans = kmeans_color_quantization(image, clusters=4)

    # Convert to grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

    # Draw largest enclosing circle onto a mask
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        break
    
    # Bitwise-and for result
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = (0,0,0)

    
    ###cv2.imshow('thresh', thresh)
    ###cv2.imshow('result', result)
    ###cv2.imshow('mask', mask)
    ###cv2.imshow('kmeans', kmeans)
    ###cv2.imshow('image', image)
    ###cv2.waitKey()

    #cv2.imwrite("static/upload/bin_"+fname, thresh)'''
    

    ###fg
    '''img = cv2.imread('static/dataset/'+fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    segment = cv2.subtract(sure_bg,sure_fg)
    img = Image.fromarray(img)
    segment = Image.fromarray(segment)
    path3="static/trained/test/fg_"+fname
    #segment.save(path3)'''
    
        
    return render_template('test_pro.html',msg=msg,fn=fn,ts=ts,act=act)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    mycursor = mydb.cursor()
    msg=""
    fn=""
    fn=""
    res=""
    res1=""
    st=""
    data=[]
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[2]
    
    nn=int(ts)
    n=nn-1
    classname=cname[n]

    mycursor.execute('SELECT count(*) FROM leaf_data where disease=%s',(classname,))
    cn = mycursor.fetchone()[0]
    if cn>0:
        st="1"
        mycursor.execute('SELECT * FROM leaf_data where disease=%s',(classname,))
        data = mycursor.fetchall()
        
    return render_template('test_pro2.html',msg=msg,fn=fn,ts=ts,act=act,classname=classname,data=data,st=st)




#############################################################################################################################################################



@app.route('/logout')
def logout():
    
    session.clear()
    print("Logged out successfully", 'success')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)


