document.addEventListener("DOMContentLoaded", async () => {
    const taskSelect = document.getElementById("task_select");
    const trialSelect = document.getElementById("trial_select");

    if (taskSelect) taskSelect.addEventListener("change", updateTrials);
    if (trialSelect) trialSelect.addEventListener("change", updateTrialData);

    updateTasks();
    refreshStats();
    setInterval(updateLogs, 2000);
    setInterval(updateOptunaLogs, 2000);
    setInterval(refreshStats, 3000);

    document.querySelectorAll('.tab').forEach(tabBtn => {
        tabBtn.addEventListener('click', () => {
            // Remove active from all buttons and content
            document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add active to clicked button and corresponding tab-content
            tabBtn.classList.add('active');
            document.getElementById(tabBtn.dataset.tab).classList.add('active');
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

async function updateTrialData() {
    
    const task = document.getElementById('task_select').value;
    const trial = document.getElementById('trial_select').value;
    if (!task || !trial) return;
    
    const resp = await fetch(`/trial_data?task_name=${task}&trial_name=${trial}`);
    const data = await resp.json();
    
    document.getElementById('trial_data').innerHTML = `
    <h3>${task} - ${trial}</h3>
    <img src="/task/${task}/${trial}/plot.png" style="max-width:100%;margin-bottom:20px;">
    ${data.table_html}
    `;

    document.getElementById('test_data').innerHTML = `
        <h2>${task} - ${trial}</h3>
        <h3>- Analysis Report -</h2>
        <img src="/task/${task}/${trial}/analysis_plot.png" style="max-width:100%;">
        <h3>- Testing -</h2>
        <img src="/task/${task}/${trial}/test.png" style="max-width:100%;">
        <h3>- Training -</h2>
        <img src="/task/${task}/${trial}/train.png" style="max-width:100%;">
    `;
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

    // ✅ rebind the event each time the DOM is updated
    const killBtn = document.getElementById("kill_btn");
    if (killBtn) {
        killBtn.onclick = async () => {
            const statusEl = document.getElementById("launch_status");
            statusEl.textContent = "🛑 Killing task...";
            try {
                const resp = await fetch("/kill_task", { method: "POST" });
                const msg = await resp.json();
                statusEl.textContent = msg.status;
                refreshStats();
            } catch (e) {
                statusEl.textContent = `[ERROR] ${e}`;
            }
        };
    }
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

document.getElementById('launch_optuna_form').addEventListener('submit', async (e)=>{
    e.preventDefault();
    const data = new FormData(e.target);
    const status = document.getElementById('launch_status');
    status.textContent = '🚧 Launching Optuna...';

    const res = await fetch('/launch_optuna', { method: 'POST', body: data });
    const json = await res.json();

    status.textContent = json.status + (json.log_file ? ` (log: ${json.log_file})` : '');

    // ✅ Start auto-updating Optuna logs
    setInterval(updateOptunaLogs, 2000);
});


async function updateLogs() {
    try {
        const task = document.getElementById('task_select').value;
        const container = document.getElementById("process_log");
        console.log(task)

        if (!task) {
            container && (container.innerHTML = "<p>No task selected.</p>");
            return;
        }

        const response = await fetch(`/logs?task_name=${task}`);
        const html = await response.text();
        container.innerHTML = html;
        container.scrollTop = container.scrollHeight;
    } catch (err) {
        console.error("Error fetching logs:", err);
    }
}

async function updateOptunaLogs() {

    try {
        const container = document.getElementById("optuna_log");

        const response = await fetch(`/logs_optuna`);
        const html = await response.text();
        container.innerHTML = html;
        container.scrollTop = container.scrollHeight;
    } catch (err) {
        console.error("Error fetching logs:", err);
    }
}

// Refresh periodically
setInterval(updateLogs,2000);
setInterval(refreshStats,3000);