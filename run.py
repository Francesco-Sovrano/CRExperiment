import json
import os
import uuid
from collections import Counter
import datetime
import sys
# import psycopg2

from flask import Flask, render_template, request, make_response, redirect, \
    url_for

from parser import Parser

# set the project root directory as the templates folder, you can set others.
app = Flask(__name__)

# All experiments are saved in the source folder 'resources/experiments'.
task1_path = os.path.join("resources", "experiments","task1")
task2_path = os.path.join("resources", "experiments","task2")
task3_path = os.path.join("resources", "experiments","task3")
task4_path = os.path.join("resources", "experiments","task4")

# The list experiments contains all the files in the experiment directory
#(_, _, experiments) = next(os.walk(experiments_path))

# Counters of how many experiments have started
experiment1_started = Counter()
experiment1_started['t1CorrMAGIX'] = 0
experiment1_started['t1CorrNonMAGIX'] = 0
experiment1_started['t1InCorrMAGIX'] = 0
experiment1_started['t1InCorrNonMAGIX'] = 0

# Counters of how many experiments have concluded
experiment1_concluded = Counter()
experiment1_concluded['t1CorrMAGIX'] = 0
experiment1_concluded['t1CorrNonMAGIX'] = 0
experiment1_concluded['t1InCorrMAGIX'] = 0
experiment1_concluded['t1InCorrNonMAGIX'] = 0

# Counters of how many experiments have started
experiment2_started = Counter()
experiment2_started['t2CorrMAGIX'] = 0
experiment2_started['t2CorrNonMAGIX'] = 0
experiment2_started['t2InCorrMAGIX'] = 0
experiment2_started['t2InCorrNonMAGIX'] = 0

# Counters of how many experiments have concluded
experiment2_concluded = Counter()
experiment2_concluded['t2CorrMAGIX'] = 0
experiment2_concluded['t2CorrNonMAGIX'] = 0
experiment2_concluded['t2InCorrMAGIX'] = 0
experiment2_concluded['t2InCorrNonMAGIX'] = 0

# Counters of how many experiments have started
experiment3_started = Counter()
experiment3_started['t3CorrMAGIX'] = 0
experiment3_started['t3CorrNonMAGIX'] = 0
experiment3_started['t3InCorrMAGIX'] = 0
experiment3_started['t3InCorrNonMAGIX'] = 0

# Counters of how many experiments have concluded
experiment3_concluded = Counter()
experiment3_concluded['t3CorrMAGIX'] = 0
experiment3_concluded['t3CorrNonMAGIX'] = 0
experiment3_concluded['t3InCorrMAGIX'] = 0
experiment3_concluded['t3InCorrNonMAGIX'] = 0

# Counters of how many experiments have started
experiment4_started = Counter()
experiment4_started['t4CorrMAGIX'] = 0
experiment4_started['t4CorrNonMAGIX'] = 0
experiment4_started['t4InCorrMAGIX'] = 0
experiment4_started['t4InCorrNonMAGIX'] = 0

# Counters of how many experiments have concluded
experiment4_concluded = Counter()
experiment4_concluded['t4CorrMAGIX'] = 0
experiment4_concluded['t4CorrNonMAGIX'] = 0
experiment4_concluded['t4InCorrMAGIX'] = 0
experiment4_concluded['t4InCorrNonMAGIX'] = 0

# Creation of log file based on id name
html_tags = ["<li", "<ul", "<a"]

p = Parser()

#added for connecting postgresql
#conn = psycopg2.connect(host='localhost', dbname='postgres', user='postgres', password="Zurich@1491", port=5432)
#conn = psycopg2.connect(host='ec2-54-246-1-94.eu-west-1.compute.amazonaws.com', dbname='d20usq666h2m0p', user='qpursjrfhybeai', password="f9755cef7d73bb0be44eab3eb8fba472f2d6481431342821c00d3e853166cf7b", port=5432)


def choose_experiment_task1():
    """
    Assign an experiment to a new user. We choose the type of experiment that
    has the least amount of concluded experiments.
    If we have more than one such case, we choose the one that has the least
    amount of started experiments.
    """
    print(experiment1_concluded)
    min_val = experiment1_concluded.most_common()[-1][1] #least common experiment1_concluded

    mins = []
    for k in experiment1_concluded:
        if experiment1_concluded[k] == min_val:
            mins.append(k)

    if len(mins) > 1:
        # more than 1 type has the same amount of concluded
        # experiments. Hence, we choose the one that has the least amount of
        # started ones
        min_val = sys.maxsize
        to_assing = ''
        for k in mins:
            if experiment1_started[k] < min_val:
                min_val = experiment1_started[k]
                to_assing = k
    else:
        to_assing = mins[0]

    print("Assigned to " + to_assing)

    if to_assing.startswith('t1CorrMAGIX'):
        cr = 'task1_corr_MAGIX.json'
    elif to_assing.startswith('t1CorrNonMAGIX'):
        cr = 'task1_corr_NonMAGIX.json'
    # elif to_assing.startswith('t1CorrNoXAI'):
    #     cr = 'task1_corr_noXAI.json'
    elif to_assing.startswith('t1InCorrMAGIX'):
        cr = 'task1_incorr_MAGIX.json'
    # elif to_assing.startswith('t1InCorrNoXAI'):
    #     cr = 'task1_incorr_noXAI.json'
    else:
        cr = 'task1_incorr_NonMAGIX.json'

    experiment1_started[to_assing] += 1
    print("cr " + cr)
    return cr

def choose_experiment_task2():
    """
    Assign an experiment to a new user. We choose the type of experiment that
    has the least amount of concluded experiments.
    If we have more than one such case, we choose the one that has the least
    amount of started experiments.
    """
    print(experiment2_concluded)
    min_val = experiment2_concluded.most_common()[-1][1] #least common experiment1_concluded

    mins = []
    for k in experiment2_concluded:
        if experiment2_concluded[k] == min_val:
            mins.append(k)

    if len(mins) > 1:
        # more than 1 type has the same amount of concluded
        # experiments. Hence, we choose the one that has the least amount of
        # started ones
        min_val = sys.maxsize
        to_assing = ''
        for k in mins:
            if experiment2_started[k] < min_val:
                min_val = experiment2_started[k]
                to_assing = k
    else:
        to_assing = mins[0]

    print("Assigned to " + to_assing)

    if to_assing.startswith('t2CorrMAGIX'):
        cr = 'task2_corr_MAGIX.json'
    elif to_assing.startswith('t2CorrNonMAGIX'):
        cr = 'task2_corr_NonMAGIX.json'
    # elif to_assing.startswith('t2CorrNoXAI'):
    #     cr = 'task2_corr_noXAI.json'
    elif to_assing.startswith('t2InCorrMAGIX'):
        cr = 'task2_incorr_MAGIX.json'
    # elif to_assing.startswith('t2InCorrNoXAI'):
    #     cr = 'task2_incorr_noXAI.json'
    else:
        cr = 'task2_incorr_NonMAGIX.json'

    experiment2_started[to_assing] += 1
    print("cr " + cr)
    return cr


def choose_experiment_task3():
    """
    Assign an experiment to a new user. We choose the type of experiment that
    has the least amount of concluded experiments.
    If we have more than one such case, we choose the one that has the least
    amount of started experiments.
    """
    print(experiment3_concluded)
    min_val = experiment1_concluded.most_common()[-1][1] #least common experiment1_concluded

    mins = []
    for k in experiment3_concluded:
        if experiment3_concluded[k] == min_val:
            mins.append(k)

    if len(mins) > 1:
        # more than 1 type has the same amount of concluded
        # experiments. Hence, we choose the one that has the least amount of
        # started ones
        min_val = sys.maxsize
        to_assing = ''
        for k in mins:
            if experiment3_started[k] < min_val:
                min_val = experiment3_started[k]
                to_assing = k
    else:
        to_assing = mins[0]

    print("Assigned to " + to_assing)

    if to_assing.startswith('t3CorrMAGIX'):
        cr = 'task3_corr_MAGIX.json'
    elif to_assing.startswith('t3CorrNonMAGIX'):
        cr = 'task3_corr_NonMAGIX.json'
    # elif to_assing.startswith('t3CorrNoXAI'):
    #     cr = 'task3_corr_noXAI.json'
    elif to_assing.startswith('t3InCorrMAGIX'):
        cr = 'task3_incorr_MAGIX.json'
    # elif to_assing.startswith('t3InCorrNoXAI'):
    #     cr = 'task3_incorr_noXAI.json'
    else:
        cr = 'task3_incorr_NonMAGIX.json'

    experiment3_started[to_assing] += 1
    print("cr " + cr)
    return cr


def choose_experiment_task4():
    """
    Assign an experiment to a new user. We choose the type of experiment that
    has the least amount of concluded experiments.
    If we have more than one such case, we choose the one that has the least
    amount of started experiments.
    """
    print(experiment4_concluded)
    min_val = experiment4_concluded.most_common()[-1][1] #least common experiment4_concluded

    mins = []
    for k in experiment4_concluded:
        if experiment4_concluded[k] == min_val:
            mins.append(k)

    if len(mins) > 1:
        # more than 1 type has the same amount of concluded
        # experiments. Hence, we choose the one that has the least amount of
        # started ones
        min_val = sys.maxsize
        to_assing = ''
        for k in mins:
            if experiment4_started[k] < min_val:
                min_val = experiment4_started[k]
                to_assing = k
    else:
        to_assing = mins[0]

    print("Assigned to " + to_assing)

    if to_assing.startswith('t4CorrMAGIX'):
        cr = 'task4_corr_MAGIX.json'
    elif to_assing.startswith('t4CorrNonMAGIX'):
        cr = 'task4_corr_NonMAGIX.json'
    # elif to_assing.startswith('t4CorrNoXAI'):
    #     cr = 'task4_corr_noXAI.json'
    elif to_assing.startswith('t4InCorrMAGIX'):
        cr = 'task4_incorr_MAGIX.json'
    # elif to_assing.startswith('t4InCorrNoXAI'):
    #     cr = 'task4_incorr_noXAI.json'
    else:
        cr = 'task4_incorr_NonMAGIX.json'

    experiment4_started[to_assing] += 1
    print("cr " + cr)
    return cr


@app.route('/')
def index():
    """
    Start of the application. It return the file "templates/index.html" and
    create the unique identifier "userid", if not already found.
    """
    resp = make_response(render_template('index.html',
                                         title="Intro"))
    user_id = request.cookies.get('experiment-userid', None)

    log_info = {
        'User-Agent': request.user_agent.string,
        'Request URL': request.url,
        'Referrer': request.referrer
    }

    log_data(str(user_id), "start", json.dumps(log_info))

    if user_id is None:
        user_id = uuid.uuid4()
        resp.set_cookie('experiment-userid', str(user_id))
        print(f'Userid was None, now is {user_id}')
    return resp


@app.route("/start", methods=['GET', 'POST'])
def start():
    """
    Loading of Personal Information Survey. These questions can be found and
    changed in "templates/initial_questions.html"
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)

    # to avoid people re-doing the experiment, we set a cookie.
    first_time = request.cookies.get('experiment-init-questions', 'first-time')

    if first_time == 'first-time':
        log_data(str(user_id), "start", "initial_questions")
        resp = make_response(render_template('initial_questions.html',
                                             title="Initial Questions"))
        return resp
    else:
        return redirect(url_for('already_done'))
    
@app.route("/pre_dem_que", methods=['GET', 'POST'])
def pre_questions():
    """
    Return the page "templates/pre_dem_questions.html"
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        honeypot = request.form.get("honeypot_field", "").strip()
        if honeypot:
            # Detected bot submission
            app.logger.warning(f"Honeypot triggered by {user_id}. Bot submission ignored.")
            return "Bot submission rejected.", 400
        
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)

    resp = make_response(render_template("pre_dem_questions.html", title='Pre experiment Questions'))
    return resp

@app.route("/instructions", methods=['GET', 'POST'])
def get_instructions():
    """
    Return the page "templates/instructions.html"
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)

    resp = make_response(render_template("instructions.html", title='Instructiosn for task'))
    return resp


@app.route("/experiment1", methods=['GET', 'POST'])
def task1():
    """
    Starts the experiment. This function calls "choose_experiment()" to decide
    which experiment to assign to the user. Afterwords, it reads the apposite
    file from "resources/experiments" and populate the page.
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)
    log_data(str(user_id), "start", "scenario1")

    # Choosing experiment
    cr_file = choose_experiment_task1()
    log_data(str(user_id), "task1", cr_file)

    exp_is_done = request.cookies.get('experiment-is_done', 'not_done')
    if exp_is_done != 'DONE':
        json_path = os.path.join(task1_path, cr_file)

        if cr_file.endswith("NoXAI.json"):
            ai_explanation = None  # or set explanation block to be hidden
        else:
            with open(json_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"JSON file is empty: {json_path}")
                
                try:
                    data = json.loads(content)
                    ai_explanation = data.get('ai_explanation')
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in file: {json_path}\n{str(e)}")

                resp = make_response(render_template('experiment1.html',
                                    ai_answer=data['ai_answer'],
                                    ai_explanation=ai_explanation))
                resp.set_cookie('experiment-experimentCRtype', cr_file)
                return resp
    else:
        return redirect(url_for('already_done'))
    

@app.route("/experiment2", methods=['GET', 'POST'])
def task2():
    """
    Starts the experiment. This function calls "choose_experiment()" to decide
    which experiment to assign to the user. Afterwords, it reads the apposite
    file from "resources/experiments" and populate the page.
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)
    log_data(str(user_id), "start", "scenario2")

    # Choosing experiment
    cr_file = choose_experiment_task2()
    log_data(str(user_id), "task2", cr_file)

    exp_is_done = request.cookies.get('experiment-is_done', 'not_done')
    if exp_is_done != 'DONE':
        json_path = os.path.join(task2_path, cr_file)

        if cr_file.endswith("NoXAI.json"):
            ai_explanation = None  # or set explanation block to be hidden
        else:
            with open(json_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"JSON file is empty: {json_path}")
                
                try:
                    data = json.loads(content)
                    ai_explanation = data.get('ai_explanation')
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in file: {json_path}\n{str(e)}")

                resp = make_response(render_template('experiment2.html',
                                    ai_answer=data['ai_answer'],
                                    ai_explanation=ai_explanation))
                resp.set_cookie('experiment-experimentCRtype', cr_file)
                return resp
    else:
        return redirect(url_for('already_done'))

@app.route("/experiment3", methods=['GET', 'POST'])
def task3():
    """
    Starts the experiment. This function calls "choose_experiment()" to decide
    which experiment to assign to the user. Afterwords, it reads the apposite
    file from "resources/experiments" and populate the page.
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)
    log_data(str(user_id), "start", "scenario3")

    # Choosing experiment
    cr_file = choose_experiment_task3()
    log_data(str(user_id), "task3", cr_file)

    exp_is_done = request.cookies.get('experiment-is_done', 'not_done')
    if exp_is_done != 'DONE':
        json_path = os.path.join(task3_path, cr_file)

        if cr_file.endswith("NoXAI.json"):
            ai_explanation = None  # or set explanation block to be hidden
        else:
            with open(json_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"JSON file is empty: {json_path}")
                
                try:
                    data = json.loads(content)
                    ai_explanation = data.get('ai_explanation')
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in file: {json_path}\n{str(e)}")

                resp = make_response(render_template('experiment3.html',
                                    ai_answer=data['ai_answer'],
                                    ai_explanation=ai_explanation))
                resp.set_cookie('experiment-experimentCRtype', cr_file)
                return resp
    else:
        return redirect(url_for('already_done'))
    
@app.route("/experiment4", methods=['GET', 'POST'])
def task4():
    """
    Starts the experiment. This function calls "choose_experiment()" to decide
    which experiment to assign to the user. Afterwords, it reads the apposite
    file from "resources/experiments" and populate the page.
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)
    log_data(str(user_id), "start", "scenario4")

    # Choosing experiment
    cr_file = choose_experiment_task4()
    log_data(str(user_id), "task4", cr_file)

    exp_is_done = request.cookies.get('experiment-is_done', 'not_done')
    if exp_is_done != 'DONE':
        json_path = os.path.join(task4_path, cr_file)

        if cr_file.endswith("NoXAI.json"):
            ai_explanation = None  # or set explanation block to be hidden
        else:
            with open(json_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"JSON file is empty: {json_path}")
                
                try:
                    data = json.loads(content)
                    ai_explanation = data.get('ai_explanation')
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in file: {json_path}\n{str(e)}")

                resp = make_response(render_template('experiment4.html',
                                    ai_answer=data['ai_answer'],
                                    ai_explanation=ai_explanation))
                resp.set_cookie('experiment-experimentCRtype', cr_file)
                return resp
    else:
        return redirect(url_for('already_done'))

@app.route("/experiment_concluded", methods=['GET', 'POST'])
def experiment_concluded():
    """
    After the experiment, you may ask the participant to answer some questions.
    This function reads the questions from "resources/post_questions.txt" and
    populate the page "templates/experiment_concluded.html"
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    exp_is_done = request.cookies.get('experiment-is_done', 'not_done')

    if request.method == 'POST':
        data: dict = request.form.to_dict()

        # print("DATA::::",data["hidden_log"]["data"], flush=True)
        # print("DATA::::",data, flush=True)

        for key in data.keys():
            if key == 'hidden_log':
                d = json.loads(data[key])
                for log in d['data']:
                    splitted = log.strip().split(";")
                    action = splitted[1]
                    info = ';'.join(splitted[2:])
                    if action == "codesugg_user":
                        user = info

        log_received_data(user_id, data)

    log_data(str(user_id), "end", "code_review_task_end")
    if exp_is_done != 'DONE':
        #post_questions = read_files("post_questions.txt")
        resp = make_response(render_template('experiment_concluded.html',
                                             #post_questions=post_questions,
                                             title="Post Questions", data=user))
        return resp
    else:
        return redirect(url_for('already_done'))


@app.route("/dem_que", methods=['GET', 'POST'])
def demographic_questions():
    """
    Return the page "templates/dem_questions.html"
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)

    resp = make_response(render_template("dem_questions.html", title='Demographic Questions'))
    return resp

@app.route("/feedback", methods=['GET', 'POST'])
def feedback():
    """
    As for the final page, we ask the participants for feedback.
    Return the page "templates/feedback.html"
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)

    resp = make_response(render_template("feedback.html", title='Feedback'))
    return resp


@app.route("/conclusion", methods=['GET', 'POST'])
def conclusion():
    """
    Finally, thank the participant.
    Return "templates/conclusion.html"
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    if request.method == 'POST':
        data: dict = request.form.to_dict()
        log_received_data(user_id, data)

    log_data(str(user_id), "end", "experiment_concluded")

    exp_type = request.cookies.get('experiment-experimentCRtype')

    # update the correspondent counter
    if exp_type == 'task1_corr_MAGIX.json':
        experiment1_concluded['t1CorrMAGIX'] += 1
    elif exp_type == 'task1_corr_NonMAGIX.json':
        experiment1_concluded['t1CorrNonMAGIX'] += 1
    elif exp_type == 'task1_corr_NoXAI.json':
        experiment1_concluded['t1CorrNoXAI'] += 1
    elif exp_type == 'task1_incorr_MAGIX.json':
        experiment1_concluded['t1InCorrMAGIX'] += 1
    elif exp_type == 'task1_incorr_NonMAGIX.json':
        experiment1_concluded['t1IncorrNonMAGIX'] += 1
    elif exp_type == 'task1_incorr_NoXAI.json':
        experiment1_concluded['t1IncorrNoXAI'] += 1
    elif exp_type == 'task2_corr_MAGIX.json':
        experiment1_concluded['t2CorrMAGIX'] += 1
    elif exp_type == 'task2_corr_NonMAGIX.json':
        experiment1_concluded['t2CorrNonMAGIX'] += 1
    elif exp_type == 'task2_corr_NoXAI.json':
        experiment1_concluded['t2CorrNoXAI'] += 1
    elif exp_type == 'task2_incorr_MAGIX.json':
        experiment1_concluded['t2InCorrMAGIX'] += 1
    elif exp_type == 'task2_incorr_NonMAGIX.json':
        experiment1_concluded['t2IncorrNonMAGIX'] += 1
    elif exp_type == 'task2_incorr_NoXAI.json':
        experiment1_concluded['t2IncorrNoXAI'] += 1
    elif exp_type == 'task3_corr_MAGIX.json':
        experiment1_concluded['t3CorrMAGIX'] += 1
    elif exp_type == 'task3_corr_NonMAGIX.json':
        experiment1_concluded['t3CorrNonMAGIX'] += 1
    elif exp_type == 'task3_corr_NoXAI.json':
        experiment1_concluded['t3CorrNoXAI'] += 1
    elif exp_type == 'task3_incorr_MAGIX.json':
        experiment1_concluded['t3InCorrMAGIX'] += 1
    elif exp_type == 'task3_incorr_NonMAGIX.json':
        experiment1_concluded['t3IncorrNonMAGIX'] += 1
    elif exp_type == 'task3_incorr_NoXAI.json':
        experiment1_concluded['t3IncorrNoXAI'] += 1
    elif exp_type == 'task4_corr_MAGIX.json':
        experiment1_concluded['t4CorrMAGIX'] += 1
    elif exp_type == 'task4_corr_NonMAGIX.json':
        experiment1_concluded['t4CorrNonMAGIX'] += 1
    elif exp_type == 'task4_corr_NoXAI.json':
        experiment1_concluded['t4CorrNoXAI'] += 1
    elif exp_type == 'task4_incorr_MAGIX.json':
        experiment1_concluded['t4InCorrMAGIX'] += 1
    elif exp_type == 'task4_incorr_NonMAGIX.json':
        experiment1_concluded['t4IncorrNonMAGIX'] += 1
    elif exp_type == 'task4_incorr_NoXAI.json':
        experiment1_concluded['t4IncorrNoXAI'] += 1

    #conclusion_text = read_files("conclusion.txt")
    return render_template("conclusion.html", title='conclusion')
                           #,conclusion=conclusion_text)

def build_experiments(experiment_snippets):
    codes = []
    for num_experiment in experiment_snippets:
        experiment_snippet = experiment_snippets[num_experiment]
        codes.append({
            "id": num_experiment,
            "filename": experiment_snippet['filename'],
            "linecount": max(experiment_snippet['num_lines_L'],
                             experiment_snippet['num_lines_R']),
            "contextLineCount": 1,
            "left_line_number": 1,
            "left_content": experiment_snippet['L'],
            "right_line_number": 1,
            "right_content": experiment_snippet['R'],
            "prefix_line_count": 1,
            "prefix_escaped": 1,
            "suffix_escaped": 1,
            "comment_user": experiment_snippet['comment_user'],
            "codesugg_user": experiment_snippet['codesugg_user'],
            "comment": experiment_snippet['comment'],
            "comment_lineNumber": experiment_snippet['comment_lineNumber'],
            "subtracted_lineNumber": experiment_snippet['subtracted_lineNumber'],
            "added_lineNumber": experiment_snippet['added_lineNumber'],
            "total_lineNumber": experiment_snippet['total_lineNumber'],
            "code_suggestion_L": experiment_snippet['code_suggestion_L'],
            "code_suggestion_R": experiment_snippet['code_suggestion_R'],
        })
    return codes


@app.route("/already_done", methods=['GET', 'POST'])
def already_done():
    """
    If a participant tries to complete the experiment twice, we send him
    directly to the conclusion.
    """
    user_id = request.cookies.get('experiment-userid', 'userNotFound')
    log_data(str(user_id), "already_done", "already_done")
    conclusion_text = "Sorry but it seems you already tried to answer the " \
                      "initial questions. <br>This means that (1) you " \
                      "failed the first time, or (2) you already did the " \
                      "experiment. In both cases, you can't take the " \
                      "experiment a second time, sorry!"
    return render_template("conclusion.html", title='Already done',
                           conclusion=conclusion_text)


def log_received_data(user_id, data):
    """
    We log all the data to a file (filename=userid)
    """
    for key in data.keys():
        if key == 'hidden_log':
            d = json.loads(data[key])
            for log in d['data']:
                splitted = log.strip().split(";")
                dt = splitted[0]
                action = splitted[1]
                info = ';'.join(splitted[2:])
                log_data(user_id, action, info)
        else:
            log_data(user_id, key, data[key])


def read_files(filename):
    with open(os.path.join("resources", filename)) as f:
        return p.parse_md(f, has_code=False)


def log_data(user_id: str, key: str, data: str, dt: datetime = None):
    date_time_obj = datetime.datetime.strptime(str(datetime.datetime.now()), '%Y-%m-%d %H:%M:%S.%f')
    with open(f'{user_id}.log', 'a') as f:
        #log_dt = dt if dt is not None else datetime.timestamp(datetime.now())
        #log_dt = dt if dt is not None else date_time_obj.time()
        log_dt = dt if dt is not None else str(datetime.datetime.now())
        f.write(f'{log_dt};'
                f'{key};'
                f'{data}\n')
        
    update_sql = """INSERT INTO experiment_logs(user_id, generated_at, logs) 
    VALUES (%s, %s, %s)
    ON CONFLICT (user_id) DO UPDATE 
    SET logs = experiment_logs.logs || '|' || excluded.logs"""

    # cur = conn.cursor() 
    # cur.execute(update_sql, (user_id, datetime.datetime.now(), f'{log_dt};'f'{key};'f'{data}\n'))

    # conn.commit()
    # cur.close()
        
# This function read the experiment content from experiments/ folder. Each
# file follow the same composition rule of the experiments.
def read_experiment(file_name):
    """
    This function read the experiment content from experiments/ folder. Each
    file follow the same composition rule of the experiments.
    """
    with open(os.path.join(experiments_path, file_name)) as file_content:
        snippets = p.parse_md(file_content, has_code=True)

    questions_experiment = file_name.replace('files', 'questions')
    body = ''
    if os.path.exists(os.path.join(experiments_path, questions_experiment)):
        with open(os.path.join(experiments_path, questions_experiment)) as \
                file_content:
            body = p.parse_md(file_content, has_code=False)
    return snippets, body


def contains_html_tags(string_to_check):
    return any(tag in string_to_check for tag in html_tags)

@app.route('/data_policy', methods=['GET', 'POST'])
def data_policy():
    """
    Before the experiment, the participant should agree with the data handling policy
    Return the page "templates/data_policy.html"
    """
    resp = make_response(render_template("data_policy.html", title='Data Policy'))
    resp.set_cookie('data_policy', 'open')
    return resp