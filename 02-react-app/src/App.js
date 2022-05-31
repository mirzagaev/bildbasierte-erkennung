import React, { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const [startCam, setStartCam] = useState(false);
  const [model, setModel] = useState();
  const threshold = 0.75;
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
    tf.engine().startScope();
    if ( stream !== undefined && stream !== null ) {
      console.log("streamt...")
      model.executeAsync(process_input(stream)).then((predicts) => {
        // console.log(predicts);

        renderDetections(predicts);

        // predicts.map((predict) => {
        //   console.log(predict);
        // });

        requestAnimationFrame(() => {
          detect(stream);
        });
        tf.engine().endScope();
      });
    }
  };

  const process_input = (video_frame) => {
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    // const tfimg = tf.browser.fromPixels(video_frame).resizeNearestNeighbor([224, 224]).toInt().expandDims();
    const expandedimg = tfimg.transpose([0,1,2]).expandDims();
    return expandedimg;
  };

  const renderDetections = (detections) => {
    const predictions = detections[0].dataSync();
    const predicted_index = detections[0].as1D().argMax().dataSync()[0];
    const predictedClass = classesDir[predicted_index];
    // const score = predictions[predicted_index];
    const score = detections[1].dataSync()[0];
    const confidence = Math.round(score * 100);
    const count = detections[5].dataSync()[0];
    const boxes = detections[4].arraySync();
    if ( score > threshold ) {
      console.log("Count: ", count);
      
      console.log(predictedClass.name);
      console.log("Score: ", confidence+"%");
    }
  };

  const onLoadedData = useCallback(() => {
    detect(webcamRef?.current?.video)
  }, [startCam]);

  return (
    <section>
      { model && (
      <>
        { startCam && (
        <>
          <div className="relative">
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
          </div>
          <div className="fixed bottom-0 z-30 w-full">
            <div className="container flex flex-wrap m-auto">
              <button
                className="w-full px-5 py-3 text-xl font-bold text-white bg-gray-500 bg-opacity-75 hover:bg-gray-600"
                onClick={() => { setStartCam(false) } }>STOP</button>
            </div>
          </div>
        </>
        )}

        { !startCam && (
        <>
          <section className = "container mx-auto bg-white" >
            <div id = "startInfo" className = "container p-20 mx-auto text-center" >
              <p id = "loadingInfo" >Komponenten sind geladen, Sie können die Anwendung starten. </p>
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
      <section className = "container mx-auto bg-white" >
        <div id = "startInfo" className = "container p-20 mx-auto text-center" >
          <p id = "loadingInfo" > Komponenten werden geladen, einen kurzen Augenblick...</p>
        </div>
      </section>
      )}
    </section>
  );
}

export default App;