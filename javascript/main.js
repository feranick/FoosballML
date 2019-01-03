async function predict(){ 

///document.getElementById('output0').innerText = "Running prediction...";
//document.getElementById("button").value= "Running prediction...";
const response = await fetch("./names.txt");
const names = await response.text();
console.log(names);

var allNames = names.replace(/['\[\]]/g,'').split(',');

document.getElementById('player1').innerText = allNames[document.FoosballML.score1.value];
document.getElementById('player2').innerText = allNames[document.FoosballML.score2.value];
document.getElementById('player3').innerText = allNames[document.FoosballML.score3.value];
document.getElementById('player4').innerText = allNames[document.FoosballML.score4.value];

const model = await tf.loadModel('./model.json');
const xp = tf.tensor([[Number(document.FoosballML.score1.value),
	Number(document.FoosballML.score2.value),
	Number(document.FoosballML.score3.value),
	Number(document.FoosballML.score4.value)]]);
const prediction = model.predict(xp);

console.log(xp.print(true));
console.log(prediction.print(true));

//document.getElementById("button").value= "Predict chances of victory";
//document.getElementById('output0').innerText = "";
document.getElementById('team1a').innerText = allNames[document.FoosballML.score3.value];
document.getElementById('team1b').innerText = allNames[document.FoosballML.score4.value];
document.getElementById('output1').innerText = (prediction.dataSync()[0]*100).toFixed(1);
document.getElementById('team2a').innerText = allNames[document.FoosballML.score2.value];
document.getElementById('team2b').innerText = allNames[document.FoosballML.score4.value];
document.getElementById('output2').innerText = (prediction.dataSync()[1]*100).toFixed(1);
document.getElementById('team3a').innerText = allNames[document.FoosballML.score2.value];
document.getElementById('team3b').innerText = allNames[document.FoosballML.score3.value];
document.getElementById('output3').innerText = (prediction.dataSync()[2]*100).toFixed(1);
document.getElementById('team4a').innerText = allNames[document.FoosballML.score1.value];
document.getElementById('team4b').innerText = allNames[document.FoosballML.score4.value];
document.getElementById('output4').innerText = (prediction.dataSync()[3]*100).toFixed(1);
document.getElementById('team5a').innerText = allNames[document.FoosballML.score1.value];
document.getElementById('team5b').innerText = allNames[document.FoosballML.score3.value];
document.getElementById('output5').innerText = (prediction.dataSync()[4]*100).toFixed(1);
document.getElementById('team6a').innerText = allNames[document.FoosballML.score1.value];
document.getElementById('team6b').innerText = allNames[document.FoosballML.score2.value];
document.getElementById('output6').innerText = (prediction.dataSync()[5]*100).toFixed(1);

var fullList = "";
for (i=0; i<allNames.length; i++) {
  fullList += i + ": " + allNames[i]+"\n";
  }

document.getElementById('fullListNames').innerText = fullList;
}


predict();
