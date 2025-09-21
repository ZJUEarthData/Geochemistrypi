# Docker Deployment

Takeaway

+ Docker in the command line
+ Docker in desktop GUI
+ Dockerfile -> Docker Image -> Docker Container

1. Search how to download **Docker** in your computer. Meanwhile, you will get a command line tool called **docker**. Use `docker --version` to check whether it exists on VSCode console.

   <img  alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/d5e2bb8b-9634-40bc-84b1-36aa0977763e">

2. Navigate to the directory where **Dockerfile** stays. **Dockerfile** has been set up and tested already so you don't need to set it manually.

   <img alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/f0be0607-5ab9-489b-8f43-5657b25f0d28">

3. Use `docker build` to make a docker image. The command is might different on Windows. No worries, just search it on the internet. In the command below, `geopi` is the name of the docker image while `test` is the tag of  the docker image. For the first time to execute the command, it needs longer time than that of mine shown below and looks slightly different. Sometimes, It would fail because the stability of the network. If it fails, just rerun the command until it works.

   <img width="990" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/724ead8b-1006-4e3d-bef2-8a5ec9c95f12">

4. Open **Docker Desktop**. Now, in **Images** section, the created docker image is in the list.

   <img width="1339" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/c6e771d1-c912-495f-98b5-41d325e77060">

   Click the **run** button as shown above to configure the docker image when you want to initialize the corresponding docker container. Then, you simple input the container name and host ports as shown below, click **run** and wait until it finishes.

   <img width="1341" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/22079af1-3868-4ab1-8e43-af6c76c9c5db">

5. Navigate to **Containers** section, you would see the corresponding docker container `geopi-testing` is activated.

   <img width="1339" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/1a4f4ad8-7391-46d3-af99-e49790afcd47">

   Now, you can click the running container, and navigate to its terminal. Use `ls` to check whether the code is inside or not.

   <img width="1337" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/1b4c91f0-f6d5-471e-a748-3cb5d754f088">

6. The difference between **Local Deployment** and **Docker Deployment** is that we have set up the commands to download related dependencies for you. Hence, you can test CLI version, frontend and backend server directly. For more manual download detail, please refer to **Local Deployment**.

   (1) **CLI version**: Use the command `python start_cli_pipeline.py` to test our command-line-interface (CLI) software.

   <img width="1338" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/95442d43-37ca-43f7-b792-3b6348eb0600">

   (2) **Backend server**: Launch the backend server as well by using `python start_dash_pipeline.py`.

   <img width="1341" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/d8765208-ff69-4b90-902e-a18e6de264be">

   Until now, you can copy the address **http://0.0.0.0:8000** shown in the above picture and open it in the browser.

   <img width="1178" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/e6378cb3-0e20-453e-b0ad-ae9f03f5210d">

   This indicates that the backend server works normally.

   Besides, you can append the suffix **/docs** to the backend server address **http://0.0.0.0:8000** to make **http://0.0.0.0:8000/docs**. Then use this address to open our API interface in the browser.

   <img width="1177" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/3246c3a5-0377-48a1-9abc-8d7834155a6e">

   (3) **Frontend server**:  This time, we need to open a external terminal by clicking in **Docker Desktop**. Because, we use frontend and backend seperation mode to construct our web portal. To utilize the functionality in the web page, you need to activate frontend and backend servers simultaneously.

   <img width="1311" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/965abf04-61fe-469f-9ac6-1bbf352bd955">

   <img width="738" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/346b56d5-7082-46f2-8abc-69b2af74d26e">

   Now Navigate to the directory **frontend**, use `yarn start` to activate the frontend server.

   <img width="899" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/3094858e-d91c-47d1-9ac5-07aa955dd553">

   you can copy the address **http://localhost:3001/** shown in the above picture and open it in the browser. And then navigate to the registration page to register firstly.

   <img width="1174" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/ecb76c89-fb42-4935-8467-beb74f03b69f">

   After registeration, you can enter the home page. Hooray!

   <img width="1174" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/fe522246-7f44-4b63-bc22-781be79e5b14">
