import * as tf from '@tensorflow/tfjs';

let speechText;
let predictOutput;
let vocab;
let vocabPath = 'https://raw.githubusercontent.com/andrewsale/MP-app.github.io/main/Tokenizer/tokenizer_dictionary.json';
let tokenizer;
let model;
let modelPath = './Model_js/model.json'
// let modelPath = 'https://raw.githubusercontent.com/andrewsale/MP-app.github.io/main/Model_js/model.json';

// load the tokenizer from json
async function loadTokenizer() {
    let tknzr = fetch(vocabPath).then(response => {
        return response.json();
    })
    return tknzr;
  }

// Load the model from json
// async function loadModel() {
//     const model = 
//     return model;
//   }


// tokenize function to convert input text to list of tokenized segments
function tokenize(text) {
    text = text.toLowerCase();
    var split_text = text.split(' ');
    var tokens = [];
    split_text.forEach(element => {
        if (tokenizer[element] != undefined) {
            tokens.push(tokenizer[element]);
          }
    });
    // create a list of slices of the list of tokens
    let i = 0;
    tokenized_text_segments = [];
    while (i+50 < Math.max(tokens.length, 100)) {
        var new_slice = tokens.slice(i,i+100);
        while (new_slice.length < 100) {
            new_slice.push(0);
          }
        tokenized_text_segments.push(new_slice);
        i = i + 50;
    }
    return tokenized_text_segments;
  }


function tokenizeSpeech() {    
    var token_segments = tokenize(speechText);
    predictOutput.innerHTML = token_segments;
  }

async function init() {
    speechText = document.getElementById('userInput').value;
    predictOutput = document.getElementById('result');

    predictOutput.innerHTML = `Loading...`;

    tokenizer = await loadTokenizer();

    predictOutput.innerHTML = `Loading......`;

    model = await tf.loadLayersModel(modelPath);
    
    predictOutput.innerHTML = `Ready!`;
}

init();