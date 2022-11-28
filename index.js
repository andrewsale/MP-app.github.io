let speechText;
let predictOutput;
let theButton;
let vocab;
let vocabPath = 'https://raw.githubusercontent.com/andrewsale/MP-app.github.io/main/Tokenizer/tokenizer_dictionary.json';
let tokenizer;
let model;
let modelPath = 'https://raw.githubusercontent.com/andrewsale/MP-app.github.io/main/Model_js/model.json';

// For some reason the L2 regularization in tf does not 
// connect to the L2 regularizer in tfjs
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
    text = text.replace(/[!"#$%&()*+,-./:;<=>?@\[\\\]\^_`{|}~\t\n]/g, '')
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


// function addSample() {
//     if (sampleSelction.value = 'NHS1'){
//         document.getElementById('userInput').value = 'The NHS has been ruined by this government. Years of Tory austerity has brought the NHS to its knees.';
//     } else if (samplesSelection.value = "rivers"){
//         document.getElementById('userInput').value = "For reasons which they could not comprehend, and in pursuance of a decision by default, on which they were never consulted, they found themselves made strangers in their own country. They found their wives unable to obtain hospital beds in childbirth, their children unable to obtain school places, their homes and neighbourhoods changed beyond recognition, their plans and prospects for the future defeated; at work they found that employers hesitated to apply to the immigrant worker the standards of discipline and competence required of the native-born worker; they began to hear, as time went by, more and more voices which told them that they were now the unwanted. On top of this, they now learn that a one-way privilege is to be established by Act of Parliament; a law which cannot, and is not intended to, operate to protect them or redress their grievances, is to be enacted to give the stranger, the disgruntled and the agent provocateur the power to pillory them for their private actions."
//     }
// }



async function init() {
    sampleSelction = document.getElementById('samples');

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