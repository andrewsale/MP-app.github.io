let speechText;
let predictOutput;
let theButton;
let vocab;
let vocabPath = 'https://raw.githubusercontent.com/andrewsale/MP-app.github.io/main/Tokenizer/tokenizer_dictionary.json';
let tokenizer;
let model;
// let modelPath = './Model_js/model.json'
let modelPath = 'https://raw.githubusercontent.com/andrewsale/MP-app.github.io/main/Model_js/model.json';

class L2 {

    static className = 'L2';

    constructor(config) {
       return tf.regularizers.l2(config)
    }
}
tf.serialization.registerClass(L2);

// load the tokenizer from json
async function loadTokenizer() {
    let tknzr = fetch(vocabPath).then(response => {
        return response.json();
    })
    return tknzr;
  }

// Load the model from json
async function loadModel() {
    const model = tf.loadLayersModel(modelPath);
    return model;
  }


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

async function predictParty() {
    const prob = tf.tidy(() => {
        speechText = document.getElementById('userInput').value;        
        var x = tokenize(speechText)
        x = model.predict(tf.tensor2d(x, [x.length, 100]));
        x = tf.mean(x);
        x = x.arraySync();
        return x
    })
    if (prob < 0.5) {
        return `<p style="color:rgb(255,100,100); font-size:150%;">We predict this is by a member of the <b>LABOUR</b> party, with probability ${(100 - prob*100).toFixed(0)}%</p>`;
    } else {
        return `<p style="color:rgb(100,100,255); font-size:150%;">We predict this is by a member of the <b>CONSERVATIVE</b> party, with probability ${(prob*100).toFixed(0)}%</p>`;
    }    
}


function predictSpeech() {     
    predictParty().then((x) => {predictOutput.innerHTML = x;});
}



async function init() {
    predictOutput = document.getElementById('result');
    
    theButton = document.getElementById("predict-btn");

    theButton.innerHTML = `Loading...`;

    tokenizer = await loadTokenizer();

    theButton.innerHTML = `Loading......`;

    model = await loadModel();

    theButton.disabled = false;
    theButton.addEventListener("click", predictSpeech);   
    
    theButton.innerHTML = `Predict! (This may take a moment...)`;    
}

init();