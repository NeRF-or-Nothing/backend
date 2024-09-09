<!-- README template based on https://github.com/othneildrew/Best-README-Template -->
<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/NeRF-or-Nothing/vidtonerf">
    <img src="pics/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">NeRF or Nothing backend core repository</h3>

  <p align="center">
    A micro-services based project in rendering novel perspectives of input videos
    utilizing neural radiance fields.
    <br />
    <a href="https://github.com/NeRF-or-Nothing/backend/wiki/Learning-Resources">
        <strong>Learn more about NeRFs »</strong>
    </a>
    <br />
    <br />
    <a href="https://github.com/NeRF-or-Nothing/backend">View Demo</a>
    ·
    <a href="https://github.com/NeRF-or-Nothing/backend/issues">Report Bug</a>
    ·
    <a href="https://github.com/NeRF-or-Nothing/backend/issues">Request Feature</a>
  </p>
</div>

## About The Project

This repository contains the backend for the (Neural Radiance Fields) NeRF-or-Nothing
web application that takes raw user video and renders a novel realistic view of the
scene they captured. Neural Radiance Fields are a new technique in novel view synthesis
that has recently reached state of the art results.

### Some Important Background About NeRFs

NeRFs operate by first taking sets of input images taken at known locations and
projecting rays from each input image via a pinhole camera model projection into
3D space. Assuming the input images are all capturing different perspectives of
the same scene these reprojected rays will intersect in the center of the scene
forming a field of light rays that produce the input images (these are the initial
radiance fields). Then a small neural network is trained to predict the intensities
and colors of light along this intersecting region in order to model the radiance
fields that must have produced the initial images. This neural network is initialized
randomly for each new scene and trained uniquely to model each captured scene.
When the training is over a neural network is trained that can predict the color
and intensity of a ray when polled at a specific angle and location in the scene.
Using this trained neural network, raytracing can be used to poll the neural network
along all the rays pointing towards a new virtual camera to take a picture from the
scene at a perspective never seen before. Important to this project is the fact that
the locations for each image are needed in order to train a NeRF, we get this data
from running structure from motion (using COLMAP) on the input video. To learn more
please visit the learning resources in the wiki.

## Gaussian Splatting Background
Gaussian splatting is a novel approach to neural scene representation that offers significant 
improvements over traditional Neural Radiance Fields (NeRFs) in terms of rendering speed and
visual quality. Like NeRFs, gaussian splatting starts with a set of input images capturing 
different perspectives of the same scene, along with their corresponding camera positions and orientations.

The key difference lies in how the scene is represented and rendered:

1. **Scene Representation**: Instead of using a neural network to model the entire scene, gaussian
   splatting represents the scene as a collection of 3D Gaussian primitives. Each Gaussian
   is defined by its position, covariance matrix (which determines its shape and orientation), and 
   appearance attributes (color and opacity).

4. **Initialization**: The process begins by running structure from motion (using tools like COLMAP) on 
   the input images to obtain initial camera parameters and a sparse point cloud. This point
   cloud is used to initialize the Gaussian primitives.

5. **Training**: The system then optimizes these Gaussians to best reproduce the input images. This 
   involves adjusting the Gaussians' positions, shapes, and appearance attributes. 
   The training process is typically faster than NeRF training and can be done end-to-end using gradient descent.

6. **Rendering**: To generate a new view, the Gaussians are projected onto the image plane of the 
virtual camera. Each Gaussian splat contributes to the final image based on its projected size, shape, 
and appearance. This process is highly parallelizable and can be efficiently implemented on GPUs, 
resulting in real-time or near-real-time rendering speeds.

7. **View-dependent Effects**: Gaussian splatting can model view-dependent effects by incorporating 
additional parameters for each Gaussian, allowing for realistic 
representation of specular highlights and reflections. If you want to take advantage of this, use .ply files, and for quick reflectionless
rendering, use .splat files.

The resulting representation is compact, efficient to render, and capable of producing high-quality novel views. 
Importantly, like NeRFs, gaussian splatting requires accurate camera positions for the input images,
 which are typically obtained through structure from motion techniques.

Gaussian splatting offers several advantages over traditional NeRFs:
- Faster training times
- Real-time or near-real-time rendering of novel views
- Better preservation of fine details and sharp edges
- More compact scene representation

To learn more about gaussian splatting and its implementation details, please refer to the learning resources in the wiki.

### General Pipeline:

1. Run Structure from motion on input video (using COLMAP implementation) to
localize the camera position in 3D space for each input frame
2. Convert the structure from motion data to Normalized Device Coordinates NDC
3. Train the NeRF (implemented with TensoRF) on the input frames and their
corresponding NDC coordinates
4. Render a new virtual "flythrough" of the scene using the trained NeRF

### Project Structure

![Full Project Structure Diagram](pics/Full_Project.png)
Since running COLMAP and TensoRF takes upwards of 30 minutes per input video, this
project utilizes RabbitMQ to queue work orders for asynchronous workers to complete
user requests. MongoDb is used to keep track of active and past user jobs. The worker
implementations are under the `NeRF`, and `colmap` folders respectively while the
central webserver is under `web-server`. For more information on how these components
communicate and how data is formatted see the READMEs within each of the
aforementioned folders.

## Getting Started

### Prerequisites

1. Have [Docker](https://www.docker.com/) installed locally
2. Have a CUDA 11.7+ Nvidia GPU (To run training) 
3. Follow the service prerequisites:
   - [go-web-server]()
   - [sfm-worker]()
   - [nerf-worker]() **IMPORTANT READ**  

### Instalation

The project should be be easy to install/run once you have completed the respective prerequisites.
The files `./docker-compose-go.yml` and `docker-compose-flask.yml` handle the setup given that you want to run
V3 or V2 of the api, respectively. 

1. Clone this repository
  ```
  git clone https://github.com/NeRF-or-Nothing/backend.git
  ```

2. Compose the backend. View indepth [instructions]()
  ```
  docker compose -f <chosen_compose_file>.yml up -d
  ```

3. Follow the [frontend](https://github.com/NeRF-or-Nothing/frontend) installation.

Once everything is running the website should be available at `localhost:5173` and a video can
be uploaded to test the application.


## Output Example

Converting the training images from the [nerf-synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi)
dataset lego example to a video then running vidtonerf produces the following result:

![Example Output](pics/example_output.gif)

## Roadmap

- ** Deployment **: The team has been mixing with the idea of deploying for numerous years now. In order to do so we need to get production ready. 
    1. More request verification
    2. Reverse proxy
    3. TSL/SSL frontend
    4. Lockdown communication
- ** Colmap **: Colmap is notoriously hard to please, and we should investigate how to make it more tolerant of user videos. See [Colmap Brainstorming]() to get started.
- ** Expand functionality **: We could possible expand into a more general purpose Deep Learning powered video app. Some possibilites:
    1. Stylized Text-to-Scene: Recent research for Text-Based Scene generation has shown crazy progress on stylized/themed scene generation
- ** Testing and Cleanup **: We can always improve our codebase by implementing further testing.
- ** CI/CD Pipelines **: Upon successful deployment we could set up dedicated testing pipelines. This would be a big stretch. For now, we could create
  workflows to ensure code quality, security, and testing coverage for lighter parts of the system.
- ** Docker Hub Image Generation **: Setting up image generation would allow for users to easily start their own instance without build hassles.

## Contributing

Contributions are what make the open source community such an amazing place to
learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and
create a pull request. Please go the the relevant repository and follow this process.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Interested in the project?

Come join our [discord server](https://discord.gg/mpcJR4FvND)!

Or, inquire at: `nerf@quicktechtime.com`

## Acknowledgments

* [TensoRF project](https://github.com/apchenstu/TensoRF)
* [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* [Original NeRF](https://github.com/bmild/nerf)
* [COLMAP](https://colmap.github.io/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/NeRF-or-Nothing/vidtonerf.svg?style=for-the-badge
[contributors-url]: https://github.com/NeRF-or-Nothing/vidtonerf/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/NeRF-or-Nothing/vidtonerf.svg?style=for-the-badge
[forks-url]: https://github.com/NeRF-or-Nothing/vidtonerf/network/members
[issues-shield]: https://img.shields.io/github/issues/NeRF-or-Nothing/vidtonerf.svg?style=for-the-badge
[issues-url]: https://github.com/NeRF-or-Nothing/vidtonerf/issues
[license-shield]: https://img.shields.io/github/license/NeRF-or-Nothing/vidtonerf.svg?style=for-the-badge
[license-url]: https://github.com/NeRF-or-Nothing/vidtonerf/blob/master/LICENSE.txt
