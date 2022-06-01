import React, { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from '@tensorflow/tfjs';
import './App.scss';

function App() {
  const webcamRef = useRef(null);
  const [startCam, setStartCam] = useState(false);
  const [model, setModel] = useState();
  const [predictions, setPredictions] = useState([]);
  const schwelle = 0.75;
  const modelURL = "http://127.0.0.1:8080/tfjs_plants_mobilenetv2_640/model.json";
  const classesDir = {
    1: {
      name: 'Seerosenblatt',
      id: 1,
    },
    2: {
      name: 'Magnolienblatt',
      id: 2,
    },
    3: {
      name: 'Laub',
      id: 3
    }
  }

  useEffect( () => {
    runModel();
  }, [webcamRef]);

  const runModel = async () => {
    setModel(await tf.loadGraphModel(modelURL));
    console.log("Modell geladen.")
  }

  const detect = (stream) => {
    if (stream !== undefined && stream !== null) {
      console.log("streamt...");
      model.executeAsync(process_input(stream)).then((predicts) => {
        renderDetections(predicts);

        requestAnimationFrame(() => {
          detect(stream);
        });
      });
    }
  }

  const process_input = (video_frame) => {
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    const expandedimg = tfimg.transpose([0,1,2]).expandDims();
    return expandedimg;
  };

  const renderDetections = (detections) => {
    setPredictions([]);
    const predicted_index = detections[0].as1D().argMax().dataSync()[0];
    const predictedClass = classesDir[predicted_index];
    const score = detections[1].dataSync()[0];
    const confidence = Math.round(score * 100);
    // const count = detections[5].dataSync()[0];
    const boxes = detections[7].arraySync();

    const media = webcamRef?.current?.video;
    const videoWidth = media.offsetWidth;
    const videoHeight = media.offsetHeight;
    
    if ( score > schwelle ) {
      const detectionObjects = []
      const bbox = [];
      const minY = boxes[0][0][0] * videoHeight;
      const minX = boxes[0][0][1] * videoWidth;
      const maxY = boxes[0][0][2] * videoHeight;
      const maxX = boxes[0][0][3] * videoWidth;

      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;

      detectionObjects.push({
        class: predicted_index,
        label: predictedClass.name,
        score: confidence,
        bbox: bbox
      })

      setPredictions(detectionObjects);
    }
  };

  const onLoadedData = () => {
    detect(webcamRef?.current?.video)
  };

  return (
    <section>
      { model && (
      <>
        <>
          <div className="relative">
            { startCam && (
              <Webcam
                audio={false}
                ref={webcamRef}
                className="w-auto mx-auto"
                screenshotFormat="image/jpeg"
                videoConstraints={{
                  width: 1280,
                  height: 720,
                  facingMode: "environment"
                }}
                onLoadedData={onLoadedData}
              />
            )}

            {predictions.map((object, index) => (
              <div
                  key={index}
                  className="bounding_boxes"
                  style={{
                      left: object.bbox[0]+'px',
                      top: object.bbox[1]+'px',
                      width: object.bbox[2]+'px',
                      height: object.bbox[3]+'px'
                  }}
              >
                  <p  className="px-3 py-2 uppercase bounding_boxes_class"
                      style={{
                          top: 0,
                          left: 0,
                          right: object.bbox[1]+'px',
                          width: object.bbox[2]+'px'
                      }}>{object.label} {object.score}%
                  </p>
              </div>
              ))
            }
          </div>
          <div className="fixed bottom-0 z-30 w-full">
            <div className="container flex flex-wrap m-auto">
              <button
                className="w-full px-5 py-3 text-xl font-bold text-white bg-gray-500 bg-opacity-75 hover:bg-gray-600"
                onClick={() => { setStartCam(false) } }>STOP</button>
            </div>
          </div>
        </>
        { !startCam && (
        <>
          <section className="container mx-auto bg-white" >
            <div id="startInfo" className="container p-20 mx-auto text-center" >
              <p id="loadingInfo" >Komponenten sind geladen, Sie können die Anwendung starten. </p>
            </div>
          </section>
          <div className="fixed bottom-0 z-30 w-full">
            <div className="container flex flex-wrap m-auto">
              <button
                className="w-full px-5 py-3 text-xl font-bold text-gray-900 bg-yellow-300 bg-opacity-75 hover:bg-yellow-300"
                onClick={ () => { setStartCam(true) } }>START</button>
            </div>
          </div>
        </>
        )}
      </>
      )}
      { !model && (
      <section className="container mx-auto bg-white" >
        <div id="startInfo" className="container p-20 mx-auto text-center" >
          <p id="loadingInfo" > Komponenten werden geladen, einen kurzen Augenblick...</p>
        </div>
      </section>
      )}
    </section>
  );
}

export default App;