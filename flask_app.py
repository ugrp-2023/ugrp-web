from flask import Flask, render_template
from flask import request, redirect, url_for
import chatbot as cb

cb.delete()
qanda={}

app = Flask(__name__)


@app.route('/')
def login():
    value = ""
    return render_template('login.html', ment = value)

@app.route('/login_check')
def login_check():
    id = request.args.get('id')
    password = request.args.get('password')

    return redirect(url_for('chat', id=id))

@app.route('/chat/<id>', methods=['GET', 'POST'])
def chat(id):
    global qanda
    # id에 해당하는 qanda가 없다면 초기화
    if id not in qanda:
        qanda[id] = {'q': [], 'a': []}

    if request.method == 'POST':
        user_input = request.form['input']
        user_input = str(user_input)
        answer = cb.doing(user_input)
        qanda[id]['q'].append(user_input)
        qanda[id]['a'].append(answer)

    return render_template('chat.html', id=id, qa=qanda[id])

@app.route('/chat/clear_conversation/<id>', methods=['POST'])
def clear_conversation(id):
    global qanda
    qanda[id] = {'q': [], 'a': []}

    return redirect(url_for('chat', id=id, qa=qanda[id]))