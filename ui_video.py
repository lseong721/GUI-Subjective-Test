# -*- coding: utf-8 -*-
import os, glob
import copy
import json
from functools import partial

import requests
import numpy as np

from PyQt5 import uic, Qt, QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

import vedo
from vedo import Plane
from vedo import *
from glob import glob
import vedo.settings as settings
settings.default_font = 'Ubuntu'
settings.use_depth_peeling = True
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import time

def save_text(save_name, data: list):
    with open(save_name, 'w') as f:
        for item in data:
            f.write(str(item) + '\n')

def load_from_npy(npy_name):
    return np.load(npy_name)

def load_audio(audio_name):
    media_content = QMediaContent(QtCore.QUrl.fromLocalFile(audio_name))
    return media_content

def load_from_npz(npy_name):
    data_mesh = np.load(npy_name)
    v1 = data_mesh['v1']
    v2 = data_mesh['v2']
    v3 = data_mesh['v3']
    f = data_mesh['f']

    v1 = (v1.reshape(v1.shape[0], -1, 3))
    v2 = (v2.reshape(v1.shape[0], -1, 3))
    v3 = (v3.reshape(v1.shape[0], -1, 3))

    v1 = v1 - v1[0].mean(0).reshape(1, 1, 3)
    v2 = v2 - v2[0].mean(0).reshape(1, 1, 3)
    v3 = v3 - v3[0].mean(0).reshape(1, 1, 3)
    return v1, v2, v3, f


class UiMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # 기본 세팅
        super(UiMainWindow, self).__init__()
        uic.loadUi("win.ui", self)

        self.setAcceptDrops(True)  # 드래그앤드롭을 통한 파일 로드를 위한 설정

        self.resize(1080, 720)
        self.bg_color = (255, 255, 255)
        self.counter = 0
        self.seq_len = 1
        self.data_counter = 0
        self.duration = 0
        self.time_init = 0
        self.screenshot_scale = 3

        self.state = False
        self.is_load = False
        self.button = None
        self.slider = None

        self.data_dir = 'Y:/workspace/SDETalk2/demo/result'
        self.render_path = os.path.join(os.getcwd(), 'render')
        self.result_path = os.path.join(os.getcwd(), 'result')
        os.makedirs(self.render_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

        self.selected_value_list1 = []
        self.selected_value_list2 = []
        self.selected_value_list3 = []
        

        self.init_GUI()

    def init_GUI(self):
        self.mainWidget = QtWidgets.QWidget(self.centralwidget)
        self.mainWidget.setObjectName("mainWidget")
        self.mainGrid.addWidget(self.mainWidget, 0, 0, 1, 1)
        self.mainWidgetLayout = QtWidgets.QGridLayout()
        self.mainWidget.setLayout(self.mainWidgetLayout)

        # add vedo viewer
        self.VTKWidget = QVTKRenderWindowInteractor(self)
        self.posePlt = vedo.Plotter(shape=(1, 3), qt_widget=self.VTKWidget, bg=(255, 255, 255)).add_shadows()
        self.camera = self.posePlt.camera
        self.mainWidgetLayout.addWidget(self.VTKWidget, 0, 0, 1, 1)

        ## add slider widget
        self.sliderWidget = QtWidgets.QWidget()
        self.sliderWidget.setObjectName("sliderWidget")
        self.mainGrid.addWidget(self.sliderWidget, 1, 0, 1, 1)
        self.sliderWidgetLayout = QtWidgets.QGridLayout()
        self.sliderWidget.setLayout(self.sliderWidgetLayout)

        ## add button widget
        self.VTK1Widget = QVTKRenderWindowInteractor(self)
        self.buttonPlt = vedo.Plotter(qt_widget=self.VTK1Widget, bg=(50, 50, 50))
        self.sliderWidgetLayout.addWidget(self.VTK1Widget, 1, 0, 1, 1)
        self.button = self.buttonPlt.add_button(self.function_button_in_vedo, 
                                                states=['Play', 'Pause'], bc=['k4', 'k4'], pos=[0.075, 0.27],
                                                font=settings.default_font, size=15)
        self.slider = self.buttonPlt.add_slider(self.function_slider_in_vedo, xmin=0, xmax=1, 
                                                pos=[[0.15, 0.5], [0.95, 0.5]], slider_width=0.2)

        # subjective test area
        self.scoreLayout = QtWidgets.QGridLayout()
        self.scoreLayout.setObjectName("scoreLayout")
        self.mainGrid.addLayout(self.scoreLayout, 0, 1, 2, 1)

        self.mainGrid.setColumnStretch(0, 4)
        self.mainGrid.setColumnStretch(1, 1)
        self.mainGrid.setRowStretch(0, 10)
        self.mainGrid.setRowStretch(1, 1)

        # add textbox
        self.titletextbox = QtWidgets.QLabel("Subjective Test")
        self.titletextbox.setMinimumSize(QtCore.QSize(0, 35))
        self.scoreLayout.addWidget(self.titletextbox, 0, 0)

        # add line edit
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setMinimumSize(QtCore.QSize(0, 35))
        self.scoreLayout.addWidget(self.lineEdit, 1, 0)

        # add bar
        self.bar = QtWidgets.QFrame()
        self.bar.setFrameShape(QtWidgets.QFrame.HLine)
        self.scoreLayout.addWidget(self.bar, 2, 0)

        # add button
        self.openButton = QtWidgets.QPushButton('Open', self)
        self.openButton.setObjectName('Open Button')
        self.openButton.setMinimumSize(QtCore.QSize(0, 35))
        self.scoreLayout.addWidget(self.openButton, 3, 0)

        # add updatable text box
        self.pathtextbox = QtWidgets.QPlainTextEdit("File Path...")
        self.scoreLayout.addWidget(self.pathtextbox, 4, 0)

        # add button
        self.renderPathButton = QtWidgets.QPushButton('Open', self)
        self.renderPathButton.setObjectName('Render Path Button')
        self.renderPathButton.setMinimumSize(QtCore.QSize(0, 35))
        self.scoreLayout.addWidget(self.renderPathButton, 5, 0)
        
        # add updatable text box
        self.textboxRenderPath = QtWidgets.QPlainTextEdit("Render Directory Path: %s" % self.render_path)
        self.scoreLayout.addWidget(self.textboxRenderPath, 6, 0)

        # add button
        self.resultPathButton = QtWidgets.QPushButton('Open', self)
        self.resultPathButton.setObjectName('Result Path Button')
        self.resultPathButton.setMinimumSize(QtCore.QSize(0, 35))
        self.scoreLayout.addWidget(self.resultPathButton, 7, 0)

        # add updatable text box
        self.textboxResultPath = QtWidgets.QPlainTextEdit("Render Directory Path: %s" % self.result_path)
        self.scoreLayout.addWidget(self.textboxResultPath, 8, 0)

        # add bar
        self.bar = QtWidgets.QFrame()
        self.bar.setFrameShape(QtWidgets.QFrame.HLine)
        self.scoreLayout.addWidget(self.bar, 9, 0)

        # add radio button
        self.groupBox1 = QtWidgets.QGroupBox("Immersiveness", self)
        hbox = QtWidgets.QHBoxLayout()
        self.radio1_1 = QtWidgets.QRadioButton("Left")
        self.radio1_2 = QtWidgets.QRadioButton("Center")
        self.radio1_3 = QtWidgets.QRadioButton("Right")
        self.radio1_1.toggled.connect(self.toggle_radio_button1)
        self.radio1_2.toggled.connect(self.toggle_radio_button1)
        self.radio1_3.toggled.connect(self.toggle_radio_button1)
        hbox.addWidget(self.radio1_1)
        hbox.addWidget(self.radio1_2)
        hbox.addWidget(self.radio1_3)
        self.groupBox1.setLayout(hbox)
        self.scoreLayout.addWidget(self.groupBox1, 10, 0)
        self.radio1_2.setChecked(True)

        # add radio button
        self.groupBox2 = QtWidgets.QGroupBox("Lip sync", self)
        hbox = QtWidgets.QHBoxLayout()
        self.radio2_1 = QtWidgets.QRadioButton("Left")
        self.radio2_2 = QtWidgets.QRadioButton("Center")
        self.radio2_3 = QtWidgets.QRadioButton("Right")
        self.radio2_1.toggled.connect(self.toggle_radio_button2)
        self.radio2_2.toggled.connect(self.toggle_radio_button2)
        self.radio2_3.toggled.connect(self.toggle_radio_button2)
        hbox.addWidget(self.radio2_1)
        hbox.addWidget(self.radio2_2)
        hbox.addWidget(self.radio2_3)
        self.groupBox2.setLayout(hbox)
        self.scoreLayout.addWidget(self.groupBox2, 11, 0)
        self.radio2_2.setChecked(True)

        # add radio button
        self.groupBox3 = QtWidgets.QGroupBox("Confidence", self)
        hbox = QtWidgets.QHBoxLayout()
        self.radio3_1 = QtWidgets.QRadioButton("1")
        self.radio3_2 = QtWidgets.QRadioButton("2")
        self.radio3_3 = QtWidgets.QRadioButton("3")
        self.radio3_1.toggled.connect(self.toggle_radio_button3)
        self.radio3_2.toggled.connect(self.toggle_radio_button3)
        self.radio3_3.toggled.connect(self.toggle_radio_button3)
        hbox.addWidget(self.radio3_1)
        hbox.addWidget(self.radio3_2)
        hbox.addWidget(self.radio3_3)
        self.groupBox3.setLayout(hbox)
        self.scoreLayout.addWidget(self.groupBox3, 12, 0)
        self.radio3_2.setChecked(True)

        # add bar
        self.bar = QtWidgets.QFrame()
        self.bar.setFrameShape(QtWidgets.QFrame.HLine)
        self.scoreLayout.addWidget(self.bar, 13, 0)

        # add button
        self.clearButton = QtWidgets.QPushButton('Next', self)
        self.clearButton.setObjectName('Next Button')
        self.clearButton.setMinimumSize(QtCore.QSize(0, 35))
        self.scoreLayout.addWidget(self.clearButton, 14, 0)

        # add button
        self.renderButton = QtWidgets.QPushButton('Render', self)
        self.renderButton.setObjectName('Render Button')
        self.renderButton.setMinimumSize(QtCore.QSize(0, 35))
        self.scoreLayout.addWidget(self.renderButton, 15, 0)

        # global short cut
        self.shortCutSend1 = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+1"), self)
        self.shortCutSend1.activated.connect(self.take_snapshot_function)
        self.renderButton.clicked.connect(self.take_snapshot_function)
        self.clearButton.clicked.connect(self.next_data_function)
        self.shortCutOpen1 = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        self.shortCutOpen1.activated.connect(self.load_data_list)
        self.openButton.clicked.connect(self.load_data_list)
        self.renderPathButton.clicked.connect(self.select_render_path)
        self.resultPathButton.clicked.connect(self.select_result_path)

        # set call back
        self.buttonPlt.add_callback('mouse click', self.function_button_in_vedo)
        self.buttonPlt.add_callback('on key press', self.function_keyboard)
        self.posePlt.add_callback('on key press', self.function_keyboard)

        # set audio
        self.media_player = QMediaPlayer(self)

        self.show_all()
        self.show()

    ###########################################################################################
    def update(self):
        if self.is_load:
            if self.state:
                self.counter += 1
            if self.counter == self.seq_len:
                self.counter = 0
                self.media_player.stop()
                self.media_player.play()

            s_idx = self.counter

            if self.counter % 2 == 1:
                self.timer.setInterval(31)
            else:
                self.timer.setInterval(33)

            self.update_slider(s_idx)
            self.update_mesh(s_idx)

    def show_all(self):
        zoom = 0.7  # 0.8
        # self.posePlt1.add(self.light1)
        self.posePlt.show(at=0, zoom=zoom, resetcam=1)
        self.posePlt.show(at=1, zoom=zoom, resetcam=1)
        self.posePlt.show(at=2, zoom=zoom, resetcam=1)
        self.posePlt.render()

        self.buttonPlt.show()

        self.timer = QtCore.QTimer()
        # self.timer.setInterval(32)  # 33msec (30FPS) 마다 반복
        self.timer.timeout.connect(self.update)  # start time out 시 연결할 함수, 기본값은 self.update
        self.timer.start()
        self.show()

    # load mesh sequence
    def load_data_list(self):
        self.data_counter = 0
        self.clear_data_function()
        options = QtWidgets.QFileDialog.Options()
        self.data_list_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a Directory', self.data_dir, options=options)
        if self.data_list_path == '':
            return

        self.data_list = sorted(glob(os.path.join(self.data_list_path, '*.npz')))
        self.data_path = self.data_list[self.data_counter]
        self.pathtextbox.setPlainText(self.data_path)

        try:
            v1, v2, v3, f = load_from_npz(self.data_path)

            media_content = load_audio(self.data_path.replace('.npz', '.wav'))
            self.media_player.setMedia(media_content)
            self.media_player.durationChanged.connect(self.get_duration)
            self.is_load = True

        except:
            print('mesh data is needed')

        self.set_mesh(v1, v2, v3, f)
        self.seq_len = v1.shape[0]

    def set_mesh(self, v_data1, v_data2, v_data3, f_data):
        self.vertices_list1 = v_data1
        self.vertices_list2 = v_data2
        self.vertices_list3 = v_data3
        print(f'[Load mesh] Vertices: {v_data1.shape} / Faces: {f_data.shape}')

        mesh_data1 = [v_data1[0], f_data]
        self.mesh1 = Mesh(mesh_data1)
        # self.mesh1.compute_normals()
        # self.mesh1.lighting(ambient=0.3, diffuse=0.3)
        self.mesh1.phong()#.color('white')
        self.posePlt.at(0).add(self.mesh1)

        mesh_data2 = [v_data2[0], f_data]
        self.mesh2 = Mesh(mesh_data2)
        # self.mesh2.compute_normals()
        self.mesh2.phong()#.color('white')
        self.posePlt.at(1).add(self.mesh2)

        mesh_data3 = [v_data3[0], f_data]
        self.mesh3 = Mesh(mesh_data3)
        # self.mesh3.compute_normals()
        self.mesh3.phong()#.color('white')
        self.posePlt.at(2).add(self.mesh3)

    # Ensure media content is loaded
    def get_duration(self):
        self.duration = self.media_player.duration()  # Duration in milliseconds
        print(f"Audio duration: {self.duration} ms")

    def update_mesh(self, s_idx):
        counter = int(s_idx % len(self.vertices_list1))
        self.mesh1.vertices(self.vertices_list1[counter])
        self.mesh2.vertices(self.vertices_list2[counter])
        self.mesh3.vertices(self.vertices_list3[counter])
        self.posePlt.render()
        self.buttonPlt.render()

    def update_slider(self, s_idx):
        slider_widget = self.slider
        slider_widget.GetRepresentation().SetValue(s_idx / self.seq_len)

    def select_render_path(self):
        options = QtWidgets.QFileDialog.Options()
        self.render_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a Directory', self.render_path, options=options)
        self.textboxRenderPath.setPlainText(self.render_path)

    def select_result_path(self):
        options = QtWidgets.QFileDialog.Options()
        self.result_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a Directory', self.result_path, options=options)
        self.textboxResultPath.setPlainText(self.result_path)

    def toggle_radio_button1(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            self.selected_value1 = radio_button.text()
            print(f"Immersiveness value: {self.selected_value1}")

    def toggle_radio_button2(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            self.selected_value2 = radio_button.text()
            print(f"Lip sync value: {self.selected_value2}")

    def toggle_radio_button3(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            self.selected_value3 = radio_button.text()
            print(f"Confidence value: {self.selected_value3}")

    def take_snapshot_function(self):
        vedo.plotter_instance = self.posePlt
        data_name = os.path.basename(self.data_path).split('.')[0]
        self.posePlt.screenshot(os.path.join(self.render_path, '%s_frame_%03d.png' % (data_name, self.counter)), scale=self.screenshot_scale)

    def clear_data_function(self):
        self.media_player.stop()
        self.posePlt.clear(at=0)
        self.posePlt.clear(at=1)
        self.posePlt.clear(at=2)

        self.radio1_1.setChecked(False)
        self.radio1_2.setChecked(False)
        self.radio1_3.setChecked(False)
        self.radio2_1.setChecked(False)
        self.radio2_2.setChecked(False)
        self.radio2_3.setChecked(False)
        self.radio3_1.setChecked(False)
        self.radio3_2.setChecked(False)
        self.radio3_3.setChecked(False)

        self.state = False
        self.counter = 0

    def next_data_function(self):
        self.selected_value_list1.append(self.selected_value1)
        self.selected_value_list2.append(self.selected_value2)
        self.selected_value_list3.append(self.selected_value3)

        self.data_counter += 1
        self.clear_data_function()
        
        if len(self.data_list) > self.data_counter:
            self.data_path = self.data_list[self.data_counter]
            self.pathtextbox.setPlainText(self.data_path)

            self.selected_value_list1 = []
            self.selected_value_list2 = []
            self.selected_value_list3 = []
        else:
            subject_name = self.lineEdit.text()
            save_list = [','.join(self.selected_value_list1), ','.join(self.selected_value_list2), ','.join(self.selected_value_list3)]
            save_name = os.path.join(self.result_path, '%s.txt' % subject_name)
            save_text(save_name, save_list)
            self.data_counter = 0
            print('[Finish] Complete subjective test')
            return


        try:
            v1, v2, v3, f = load_from_npz(self.data_path)

            media_content = load_audio(self.data_path.replace('.npz', '.wav'))
            self.media_player.setMedia(media_content)
            self.media_player.durationChanged.connect(self.get_duration)
            self.is_load = True
        except:
            print('mesh data is needed')

        self.set_mesh(v1, v2, v3, f)
        self.seq_len = v1.shape[0]


    # functions
    def function_button_in_vedo(self, event):
        if self.buttonPlt.picker.GetActor2D() != self.button.actor:
            return
        self.button.switch()
        self.state = not self.state
        if self.state:
            self.media_player.play()
            self.time_init = time.time()
        else:
            self.media_player.pause()

    def function_slider_in_vedo(self, widget, event):
        if not self.state:
            s_idx = int(widget.GetRepresentation().GetValue())
            self.update_mesh(s_idx)
            self.counter = s_idx - self.seq_len * 0

            if self.counter == 0:
                self.media_player.stop()
                self.media_player.play()

    def function_keyboard(self, event=None):
        if event.keypress == 'space':
            self.button.switch()
            self.state = not self.state
            if self.state:
                self.media_player.play()
            else:
                self.media_player.pause()

        if event.keypress == 's':
            self.button.switch()
            self.state = not self.state
            self.take_snapshot_function()
