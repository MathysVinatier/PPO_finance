document.addEventListener("DOMContentLoaded", () => {
    const tabs = document.querySelectorAll(".tab");
    const contents = document.querySelectorAll(".tab-content");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            contents.forEach(c => c.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById(tab.dataset.tab).classList.add("active");
        });
    });
});

async function fetchJSON(url){return (await fetch(url)).json()}

async function updateTasks(){
    const tasks = await fetchJSON('/tasks');
    const taskSel=document.getElementById('task_select');
    taskSel.innerHTML='';
    tasks.forEach(t=>{
        const o=document.createElement('option');o.value=o.text=t;taskSel.appendChild(o);
    });
    updateTrials();
}

async function updateTrials(){
    const task=document.getElementById('task_select').value;
    const trials=await fetchJSON(`/trials/${task}`);
    const trialSel=document.getElementById('trial_select');
    trialSel.innerHTML='';
    trials.forEach(t=>{
        const o=document.createElement('option');o.value=o.text=t;trialSel.appendChild(o);
    });
    updateTrialData();
}

async function updateTrialData(){
    const task=document.getElementById('task_select').value;
    const trial=document.getElementById('trial_select').value;
    if(!task || !trial) return;
    const resp=await fetch(`/trial_data?task_name=${task}&trial_name=${trial}`);
    const data=await resp.json();
    document.getElementById('trial_data').innerHTML=
        `<h3>${task} - ${trial}</h3>
        <img src="/task/${task}/${trial}/plot.png" style="max-width:100%;margin-bottom:20px;">
        ${data.table_html}`;
}

async function refreshStats() {
    const res = await fetch("/system_stats");
    const data = await res.json();
    document.querySelector("#stats").innerHTML = `
        <p>🧠 CPU : ${data.cpu}%</p>
        <p>💾 RAM : ${data.ram}%</p>
        <p>🌡 TEMP : ${JSON.stringify(data.temps)}</p>
        <p>🎯 Active Task : <strong>${data.active_task}</strong></p>
        <button id="kill_btn" ${data.is_running ? "" : "disabled"}>🛑 Kill Task</button>
    `;
    document.querySelector("#kill_btn").onclick = async () => {
    const statusEl = document.getElementById("launch_status"); // reuse same status display
    statusEl.textContent = "🛑 Killing task...";
    try {
        const resp = await fetch("/kill_task", { method: "POST" });
        const msg = await resp.json();
        statusEl.textContent = msg.status;   // show result inline
        refreshStats();                       // update dashboard
    } catch (e) {
        statusEl.textContent = `[ERROR] ${e}`;
    }
};
}

// Launch Task
document.getElementById('launch_form').addEventListener('submit', async (e)=>{
    e.preventDefault();
    const data=new FormData(e.target);
    const params=new URLSearchParams(data);
    document.getElementById('launch_status').textContent='🚧 Launching...';
    const r=await fetch('/launch_task',{method:'POST',body:params});
    const res=await r.json();
    document.getElementById('launch_status').textContent=res.status;
});

async function updateLogs() {
    try {
        const response = await fetch("/logs");
        const html = await response.text();
        const container = document.getElementById("process_log");
        container.innerHTML = html || "<p>No logs yet.</p>";
        container.scrollTop = container.scrollHeight;
    } catch (err) {
        console.error("Error fetching logs:", err);
    }
}

// Refresh periodically
setInterval(updateLogs,2000);
setInterval(refreshStats,3000);
task_select?.addEventListener('change',updateTrials);
trial_select?.addEventListener('change',updateTrialData);
updateTasks();
refreshStats();