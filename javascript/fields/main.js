async function predict(){

   allNames = await getNames();
   listNames(allNames);

   s1 = Number(document.FoosballML.score1.value);
   s2 = Number(document.FoosballML.score2.value);
   s3 = Number(document.FoosballML.score3.value);
   s4 = Number(document.FoosballML.score4.value);

   for(var i = 1; i <= 4; i++) {
      document.getElementById("player"+i).innerText = allNames[this['s'+i]];
      }

   const model = await tf.loadLayersModel('./model.json');
   const xp = tf.tensor([[s1,s2,s3,s4]]);

   const prediction = model.predict(xp);

   console.log(xp.print(true));
   console.log(prediction.print(true));

   document.getElementById('team1a').innerText = allNames[document.FoosballML.score3.value];
   document.getElementById('team1b').innerText = allNames[document.FoosballML.score4.value];
   document.getElementById('team2a').innerText = allNames[document.FoosballML.score2.value];
   document.getElementById('team2b').innerText = allNames[document.FoosballML.score4.value];
   document.getElementById('team3a').innerText = allNames[document.FoosballML.score2.value];
   document.getElementById('team3b').innerText = allNames[document.FoosballML.score3.value];
   document.getElementById('team4a').innerText = allNames[document.FoosballML.score1.value];
   document.getElementById('team4b').innerText = allNames[document.FoosballML.score4.value];
   document.getElementById('team5a').innerText = allNames[document.FoosballML.score1.value];
   document.getElementById('team5b').innerText = allNames[document.FoosballML.score3.value];
   document.getElementById('team6a').innerText = allNames[document.FoosballML.score1.value];
   document.getElementById('team6b').innerText = allNames[document.FoosballML.score2.value];

   for(var i = 1; i <= 6; i++) {
      document.getElementById('output'+i).innerText = (prediction.dataSync()[i-1]*100).toFixed(1);
      document.getElementById('outputR'+i).innerText = (prediction.dataSync()[i-1]*100/(prediction.dataSync()[i-1]+prediction.dataSync()[6-i])).toFixed(1); }

}

async function getNames() {
    const response = await fetch("./names.txt");
    const names = await response.text();
    var allNames = names.replace(/['\[\]]/g,'').split(',');
    return allNames;
    }

function listNames(allNames) {
    var fullList = "";
    for (i=0; i<allNames.length; i++) {
        fullList += i + ": " + allNames[i]+"\n";
        }
    console.log(fullList);
    document.getElementById('fullListNames').innerText = fullList;
    }

predict();
