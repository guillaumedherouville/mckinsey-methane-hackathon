class GUI:
    """
    This class is dedicated to manage to user interface of the website. 
    It contains methods to edit the sidebar for the selected application as well as the front page.
    """

    def __init__(self):

        self.list_of_apps = [
            "Empty",
            "Object Detection",
            "Face Detection",
            "Fire Detection",
        ]
        self.guiParam = {}

    # ----------------------------------------------------------------

    def getGuiParameters(self):
        self.common_config()
        self.appDescription()
        return self.guiParam

    # ------------------------------------a----------------------------

    def common_config(self, title="Methane detection app"):  # (Beta version :golf:)
        """
        User Interface Management: Sidebar
        """
        st.image("./media/logo_inveesion.png", "InVeesion.", width=50)

        st.title(title)

        st.sidebar.markdown("### :arrow_right: Settings")

        self.dataSource = st.sidebar.radio(
            "Please select the source of your image :",
            ["Demo", "Upload", "TBD"],
        )

        # Get the application from the GUI
        self.selectedApp = st.sidebar.selectbox(
            "Chose an AI Application", self.list_of_apps
        )

        if self.selectedApp is "Empty":
            st.sidebar.warning("Select an application from the list")

        self.displayFlag = st.sidebar.checkbox("Display Real-Time Results", value=True)

        # Update the dictionnary
        self.guiParam.update(
            dict(
                selectedApp=self.selectedApp,
                dataSource=self.dataSource,
                displayFlag=self.displayFlag,
            )
        )

    # -------------------------------------------------------------------------

    def appDescription(self):

        st.header(" :arrow_right: Application: {}".format(self.selectedApp))

        if self.selectedApp == "Object Detection":
            st.info(
                "This application performs object detection using advanced deep learning models. It can detects more than 80 object from COCO dataset."
            )
            self.sidebarObjectDetection()

        elif self.selectedApp == "Face Detection":
            st.info(
                "This application performs face detection using advanced deep learning models. It can detects face in the image"
            )
            self.sidebarFaceDetection()

        elif self.selectedApp == "Fire Detection":
            st.info(
                "This application performs fire detection using advanced deep learning models. "
            )
            self.sidebarFireDetection()

        else:
            st.info(
                "To start using InVeesion dashboard you must first select an Application from the sidebar menu other than Empty"
            )

    # --------------------------------------------------------------------------
    def sidebarEmpty(self):
        pass

    # --------------------------------------------------------------------------

    def sidebarFaceDetection(self):
        """ """

        # st.sidebar.markdown("### :arrow_right: Model")
        # --------------------------------------------------------------------------
        model = st.sidebar.selectbox(
            label="Select the model",
            options=("res10_300x300_ssd_iter_140000", "opencv_face_detector"),
        )

        st.sidebar.markdown("### :arrow_right: Parameters")
        # --------------------------------------------------------------------------
        confThresh = st.sidebar.slider(
            "Confidence", value=0.80, min_value=0.0, max_value=1.00, step=0.05
        )

        self.guiParam.update(dict(confThresh=confThresh, model=model))

    # --------------------------------------------------------------------------

    def sidebarObjectDetection(self):

        # st.sidebar.markdown("### :arrow_right: Model")
        # ------------------------------------------------------#
        model = st.sidebar.selectbox(
            label="Select the model",
            options=["Caffe-MobileNetSSD", "Darknet-YOLOv3-tiny", "Darknet-YOLOv3"],
        )

        # ------------------------------------------------------#
        confThresh = st.sidebar.slider(
            "Confidence", value=0.3, min_value=0.0, max_value=1.0
        )
        nmsThresh = st.sidebar.slider(
            "Non-maximum suppression",
            value=0.30,
            min_value=0.0,
            max_value=1.00,
            step=0.05,
        )

        self.guiParam.update(
            dict(
                confThresh=confThresh,
                nmsThresh=nmsThresh,
                model=model,
                #   desired_object=desired_object
            )
        )

    # --------------------------------------------------------------------------

    def sidebarFireDetection(self):

        # st.sidebar.markdown("### :arrow_right: Model")
        # ------------------------------------------------------#
        model = st.sidebar.selectbox(
            label="Select the model", options=["Darknet-YOLOv3-tiny"]
        )

        # st.sidebar.markdown("### :arrow_right: Model Parameters")
        # ------------------------------------------------------#
        confThresh = st.sidebar.slider(
            "Confidence", value=0.5, min_value=0.0, max_value=1.0
        )
        nmsThresh = st.sidebar.slider(
            "Non-maximum suppression",
            value=0.30,
            min_value=0.0,
            max_value=1.00,
            step=0.05,
        )

        self.guiParam.update(
            dict(confThresh=confThresh, nmsThresh=nmsThresh, model=model)
        )


# ------------------------------------------------------------------
# ------------------------------------------------------------------


class DataManager:
    """ """

    def __init__(self, guiParam):
        self.guiParam = guiParam

        self.url_demo_images = {
            "NY-City": "https://s4.thingpic.com/images/8a/Qcc4eLESvtjiGswmQRQ8ynCM.jpeg",
            "Paris-street": "https://www.discoverwalks.com/blog/wp-content/uploads/2018/08/best-streets-in-paris.jpg",
            "Paris-street2": "https://www.discoverwalks.com/blog/wp-content/uploads/2018/08/best-streets-in-paris.jpg",
        }

        self.demo_image_examples = {
            "Family-picture": "./data/family.jpg",
            "Fire": "./data/fire.jpg",
            "Dog": "./data/dog.jpg",
            "Crosswalk": "./data/demo.jpg",
            "Cat": "./data/cat.jpg",
            "Car on fire": "./data/car_on_fire.jpg",
        }

        self.image = None
        self.data = None

    #################################################################
    #################################################################

    def load_image_source(self):
        """ """
        if self.guiParam["dataSource"] == "Image: Demo":

            @st.cache(allow_output_mutation=True)
            def load_image_from_path(image_path):
                image = cv.imread(image_path, cv.IMREAD_COLOR)
                # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                return image

            file_path = st.text_input("Enter the image PATH")

            if os.path.isfile(file_path):
                self.image = load_image_from_path(image_path=file_path)

            elif file_path is "":
                file_path_idx = st.selectbox(
                    "Or select a demo image from the list",
                    list(self.demo_image_examples.keys()),
                )
                file_path = self.demo_image_examples[file_path_idx]

                self.image = load_image_from_path(image_path=file_path)
            else:
                raise ValueError("[Error] Please enter a valid image path")

            # --------------------------------------------#
            # --------------------------------------------#

        elif self.guiParam["dataSource"] == "Image: Upload":

            @st.cache(allow_output_mutation=True)
            def load_image_from_upload(file):
                tmp = np.fromstring(file.read(), np.uint8)
                return cv.imdecode(tmp, 1)

            file_path = st.file_uploader("Upload an image", type=["png", "jpg"])

            if file_path is not None:
                self.image = load_image_from_upload(file_path)
            elif file_path is None:
                raise ValueError("[Error] Please upload a valid image ('png', 'jpg')")
            # --------------------------------------------#
            # --------------------------------------------#

        else:
            raise ValueError("Please select one source from the list")

        return self.image
