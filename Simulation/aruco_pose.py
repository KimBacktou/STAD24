#!/usr/bin/env python
import cv2 as cv
from cv2 import aruco
import numpy as np
import time, math
import gi
from multiprocessing import shared_memory


gi.require_version('Gst', '1.0')
from gi.repository import Gst

shared_memory_name = "aruco_shared_memory"

#Size should be the amount of values to share plus 1 for flag (x,y,z,yaw,flag). 
#Flag 0=no target, 1= found target, 2 = shutdown
array_size = 5
shared_memory_size = array_size * np.float64().nbytes  


#-------estimatePoseSingleMarkers is not available in openCV 4.9 so this is only to compensate for previous versions.
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]], 
        dtype=np.float32
    )

    rvecs = []
    tvecs = []

    for c in corners:
        _, rvec, tvec = cv.solvePnP(marker_points, c, mtx, distortion, False, cv.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs

#-----------used to get video stream from simulation
class Video():
    def __init__(self, port=5600):
        Gst.init(None)
        self.port = port
        self._frame = None
        self.video_source = 'udpsrc port={}'.format(self.port)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        return self._frame

    def frame_available(self):
        return self._frame is not None

    def run(self):
        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame
        return Gst.FlowReturn.OK

#----------
class ArucoSingleTracker:
    def __init__(self, id_to_find, marker_size, camera_size=[640, 640], show_video=False, simulation=False):
        self.id_to_find = id_to_find
        self.marker_size = marker_size
        self._show_video = show_video
        self.camera_size = camera_size
        self.is_detected = False
        self._kill = False

          # Create shared memory
        self.shared_mem = shared_memory.SharedMemory(name=shared_memory_name, create=True, size=shared_memory_size)
        self.shared_array = np.ndarray((array_size,), dtype=np.float64, buffer=self.shared_mem.buf)


        #--- 180 deg rotation matrix around the x axis
        self._R_flip = np.eye(3, dtype=np.float32)
        self._R_flip[1, 1] = -1.0
        self._R_flip[2, 2] = -1.0

        #--- Capture the camera or use simulation camera.
        if simulation:
            self.video = Video()
        else:
            self.video = cv.VideoCapture(-1)

        #---Basic readings
        self._t_read = time.time()
        self._t_detect = self._t_read
        self.fps_read = 0.0
        self.fps_detect = 0.0
    
    
    def stop(self):
        self._kill = True
        # Release shared memory
        try:
            if self.shared_mem:
                self.shared_mem.close()
                self.shared_mem.unlink()
        except Exception as e:
            print("Error while releasing shared memory:", e)


    #---Define the rotation matrix in Euler angles, well-known function.
    def _rotationMatrixToEulerAngles(self, R):
        def isRotationMatrix(R):
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            I = np.identity(3, dtype=R.dtype)
            n = np.linalg.norm(I - shouldBeIdentity)
            return n < 1e-6

        assert (isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    def _update_fps_read(self):
        t = time.time()
        self.fps_read = 1.0 / (t - self._t_read)
        self._t_read = t

    def _update_fps_detect(self):
        t = time.time()
        self.fps_detect = 1.0 / (t - self._t_detect)
        self._t_detect = t

    #----Track the markers
    def track(self, loop=True, verbose=False, show_video=None, simulation=False):
        self._kill = False
        
        if show_video is None:
            show_video = self._show_video

        marker_found = False
        x = y = z = 0

        while not self._kill:

            #------ at start no Aruco markers i found, we dont need the client to process when noting is found.
           # self.shared_array[-1] = 0
            
            #------Used to enable simulation camera or user camera
            if simulation:
                if not self.video.frame_available():
                    continue
                frame = self.video.frame().copy()  # Make a writable copy of the frame
            else:
                ret, frame = self.video.read()
                if not ret:
                    continue

            self._update_fps_read()
            #---Create a gray frame
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            #---set up the markers to use and parameters
            detector = aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=parameters)

            #---Detect markers, predefined function in the aruco lib.
            marker_corners, ids, _ = detector.detectMarkers(gray_frame)

            #Track only valid markers. 
            if ids is not None and self.id_to_find in ids:
                marker_found = True
                self._update_fps_detect()

                #--------estimate pose of markers found
                rvecs, tvecs = my_estimatePoseSingleMarkers(marker_corners, self.marker_size, camera_matrix, camera_distortion)

                #---------Only need to track our wanted marker (the marker with correct id)
                idx = np.where(ids == self.id_to_find)[0][0]
                rvec = rvecs[idx]
                tvec = tvecs[idx]

                x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]

                #----------Draw markers found
                for marker_corner, marker_id in zip(marker_corners, ids):
                    if marker_id == self.id_to_find:
                        
                        
                        #Only need to use graphics if we show a video stream
                        if show_video:
                            cv.polylines(frame, [marker_corner.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)
                            corners = marker_corner.reshape(4, 2).astype(int)
                            top_right = corners[0].ravel()
                            bottom_right = corners[2].ravel()
                            cv.drawFrameAxes(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

                        #--Use Rodrigues rotation
                        R_ct = np.matrix(cv.Rodrigues(rvec)[0])
                        R_tc = R_ct.T

                        tvec_mat = np.reshape(tvec, (3, 1))

                        #---Get marker's attitude in terms of roll, pitch, yaw
                        roll_marker, pitch_marker, yaw_marker = self._rotationMatrixToEulerAngles(self._R_flip @ R_tc)
                        yaw = math.degrees(yaw_marker)

                                        # Update shared memory
                        self.shared_array[0] = x
                        self.shared_array[1] = y
                        self.shared_array[2] = z
                        self.shared_array[3] = yaw
                        self.shared_array[-1] = 1
                        
                        #---Get camera position in x, y, z
                        pos_camera = -R_tc @ tvec_mat

                        if verbose:
                            print(f"Marker X = {tvec[0][0]:.1f}  Y = {tvec[1][0]:.1f}  Z = {tvec[2][0]:.1f} fps = {self.fps_detect:.0f}  Flag: {self.shared_array[-1]}")

                        if show_video:
                            font = cv.FONT_HERSHEY_PLAIN
                            font_color = (0, 0, 255)

                            cv.putText(frame, f"id: {marker_id[0]} Dist: {round(pos_camera[2, 0], 2)}", top_right, font, 1.3, font_color, 2, cv.LINE_AA)
                            str_position = f"x={x:.0f}  y={y:.0f}  z={z:.0f}"
                            cv.putText(frame, str_position, bottom_right, font, 1.3, font_color, 2, cv.LINE_AA)
                            str_attitude = f"MARKER Attitude roll={math.degrees(roll_marker):.0f} pitch={math.degrees(pitch_marker):.0f} yaw={math.degrees(yaw_marker):.0f}"
                            cv.putText(frame, str_attitude, (0, 20), font, 1.0, font_color, 2, cv.LINE_AA)
                            str_position = f"CAMERA Position x={pos_camera[0, 0]:.0f}  y={pos_camera[1, 0]:.0f}  z={pos_camera[2, 0]:.0f}"
                            cv.putText(frame, str_position, (0, 40), font, 1, font_color, 2, cv.LINE_AA)

            else:
                if verbose:
                    self.shared_array[-1] = 0
                    print(f"Nothing detected - fps = {self.fps_read:.0f} Flag = {self.shared_array[-1]}")
                

            if show_video:
                cv.imshow('frame', frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.shared_array[-1] = -1
                    print(f"Shutingdown Flag = {self.shared_array[-1]}")
                    time.sleep(2)
                    break
                    

        if not loop:
            return marker_found, x, y, z
        
        if simulation:
            self.video.video_pipe.set_state(Gst.State.NULL)

        else:
            print("Release memory")
            self.stop()
            self.video.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    id_to_find = 0  # Valid marker to find
    marker_size = 18  # cm
    # Shared memory setup

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # Choose dictionary for valid markers
    parameters = aruco.DetectorParameters()

    calib_data_path = "/home/STAD/Downloads/STAD-main/Simulation/MultiMatrix.npz"
    calib_data = np.load(calib_data_path)

    camera_matrix = calib_data["camMatrix"]
    camera_distortion = calib_data["distCoef"]

    aruco_tracker = ArucoSingleTracker(
        id_to_find=id_to_find,
        marker_size=marker_size,
        show_video=True,        # Enable/disable show video
        simulation=False         # Simulation or other camera
    )

    aruco_tracker.track(verbose=True,simulation=False)
