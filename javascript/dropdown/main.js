async function predict(){
    allNames = await getNames();
    listNames(allNames);

    for(var i = 1; i <= 4; i++) {
       this["p"+i] = document.getElementById("player"+i).selectedIndex;
       }

    const model = await tf.loadModel('./model.json');

    const xp = tf.tensor([[p1, p2, p3, p4]]);
    const prediction = model.predict(xp);

    console.log(xp.print(true));
    console.log(prediction.print(true));

    document.getElementById('team1a').innerText = allNames[document.FoosballML.player3.selectedIndex];
    document.getElementById('team1b').innerText = allNames[document.FoosballML.player4.selectedIndex];
    document.getElementById('team2a').innerText = allNames[document.FoosballML.player2.selectedIndex];
    document.getElementById('team2b').innerText = allNames[document.FoosballML.player4.selectedIndex];
    document.getElementById('team3a').innerText = allNames[document.FoosballML.player2.selectedIndex];
    document.getElementById('team3b').innerText = allNames[document.FoosballML.player3.selectedIndex];
    document.getElementById('team4a').innerText = allNames[document.FoosballML.player1.selectedIndex];
    document.getElementById('team4b').innerText = allNames[document.FoosballML.player4.selectedIndex];
    document.getElementById('team5a').innerText = allNames[document.FoosballML.player1.selectedIndex];
    document.getElementById('team5b').innerText = allNames[document.FoosballML.player3.selectedIndex];
    document.getElementById('team6a').innerText = allNames[document.FoosballML.player1.selectedIndex];
    document.getElementById('team6b').innerText = allNames[document.FoosballML.player2.selectedIndex];

    for(var i = 1; i <= 6; i++) {
       document.getElementById('output'+i).innerText = (prediction.dataSync()[i-1]*100).toFixed(1);
       document.getElementById('outputR'+i).innerText = (prediction.dataSync()[i-1]*100/(prediction.dataSync()[i-1]+prediction.dataSync()[6-i])).toFixed(1); 
      }
    }

async function setupMenus() {
    allNames = await getNames();
    for(var i = 1; i <= 4; i++) {
        creatPlayerMenu("player", i, allNames);
        menu = document.getElementById("player"+i);
        menu.options[i-1].selected = true;
        }
    }

async function getNames() {
    const response = await fetch("./names.txt");
    const names = await response.text();
    var allNames = names.replace(/['\[\]]/g,'').split(',');
    return allNames;
    }

function creatPlayerMenu(menuName, order, allNames){
    menu = document.getElementById(menuName+order);
    for(var i = 0; i < allNames.length; i++) {
        var opt = allNames[i];
        var el = document.createElement("option");
        el.textContent = opt;
        el.value = opt;
        //el.setAttribute("id",opt+order);
        menu.appendChild(el);
    }
}

function listNames(allNames) {
    var fullList = "";
    for (i=0; i<allNames.length; i++) {
        fullList += i + ": " + allNames[i]+"\n";
        }
    console.log(fullList);
    //document.getElementById('fullListNames').innerText = fullList;
    }

setupMenus();
predict();
