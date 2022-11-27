import * as tf from '@tensorflow/tfjs';

let model;
let modelPath = 'Model_js/model.json'
let vocab;
let vocabPath = 'Tokenizer/tokenizer_dictionary.json'
let tokenizer;
let speechText;
let predictOutput;


// Load the model from json
async function loadModel() {
  const model = await tf.loadLayersModel(modelPath);
  return model;
}

// load the tokenizer from json
async function loadTokenizer() {
  let tokenizer = await (await fetch(vocabPath)).json();
  return tokenizer;
}

// tokenize function to convert input text to list of tokenized segments
function tokenize(text) {
  text = text.toLowerCase();
  var split_text = text.split(' ');
  var token_text = [];
  split_text.array.forEach(element => {
    if (tokenizer[element] != undefined) {
      token_text.push(tokenizer[element]);
    }
  });
  // create a list of slices of the list of tokens
  let i = 0;
  tokenized_text_segments = [];
  while (i+50 < token_text.length) {
    const new_slice = token_text.slice(i,Math.min(i+100, token_text.length);
    while (new_slice.length < 100) {
      new_slice.push(0);
    }
    tokenized_text_segments.push(new_slice);
    i = i + 50;
  }
  return tokenized_text_segments;
}

async function predictParty() {
  const predictedLabel = tf.tidy(() => {
    const text = document.getElementById('user-input');
    const tokenized_text_segments = tokenize(text);
    const preds = model.predict(tf.tensor2d(tokenized_text_segments, [tokenized_text_segments.length, 100]))
    return preds.as1D().reduce((cumsum, next) => cumsum + next) / preds.length;
  }
}


function createButton(innerText, id, listener, selector, disabled = false) {
    const btn = document.createElement('BUTTON');
    btn.innerText = innerText;
    btn.id = id;
    btn.disabled = disabled;
  
    btn.addEventListener('click', listener);
    document.querySelector(selector).appendChild(btn);
}
  

function setupButtons() {
    // Button to predict
    createButton('Submit!', 'submit-btn',
      () => {
        predictParty(speechText).then((answers) => {
          // Write the answers to the output div as an unordered list.
          // It uses map create a new list of the answers while adding the list tags.
          // Then, we use join to concatenate the answers as an array with a line break
          // between answers.
          const answersList = answers.map((answer) => `<li>${answer.text} (confidence: ${answer.score})</li>`)
            .join('<br>');
  
          predictOutput.innerHTML = `<ul>${answersList}</ul>`;
        }).catch((e) => console.log(e));
      }, '#submit-button', true);
    // temp button to test tokenizer
    createButton('tokenize!', 'tokenize-btn',
      () => {
        tokenize(speechText).then((tokens) => {
          predictOutput.innerHTML = `${tokens}`;
        })
      }, '#tok-button', true)  
  }

function predict() {
  return predictParty();
}

function doTokenize() {
  return tokenize(speechText);
}

function printSpeech() {
  predictOutput.innerText = speechText;
}

async function init() {
  setupButtons();
  speechText = document.getElementById('user-input').value;
  predictOutput = document.getElementById('result');

  // model = await loadModel();
  tokenizer = await loadTokenizer();
  document.getElementById('submit-btn').disabled = false;
  document.getElementById('tokenize-btn').disabled = false;
}

init();
  