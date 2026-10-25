"""The sign-in and create-account page. Served to anyone who is not signed in."""

# __SIGNUP__ and __TEMP__ are replaced by the server with true or false.
LOGIN_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in - Complaint Analyzer</title>
<style>
:root{--bg:#DDE2E6;--panel:#FBFBFA;--ink:#16202B;--muted:#5D6B77;--line:#C4CCD3;--soft:#E2E7EA;
--bad:#B23A2B;--badbg:#F8ECEA;--sans:"Segoe UI",system-ui,-apple-system,sans-serif;--mono:Consolas,"SF Mono",monospace}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);font-family:var(--sans);
font-size:15px;line-height:1.55;display:flex;align-items:center;justify-content:center;padding:24px 16px}
.card{width:100%;max-width:420px}
.brand{border-bottom:2px solid var(--ink);padding-bottom:12px;margin-bottom:18px}
h1{margin:0;font-size:26px;letter-spacing:-.02em;line-height:1.15}
.brand p{margin:4px 0 0;color:var(--muted);font-size:13.5px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:20px}
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin:-20px -20px 18px;padding:0 8px}
.tabs button{background:none;border:none;border-bottom:2px solid transparent;margin-bottom:-1px;
padding:12px 14px;font:inherit;font-weight:500;color:var(--muted);cursor:pointer}
.tabs button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--ink)}
.tabs button:hover{color:var(--ink)}
label{display:block;font-size:12.5px;color:var(--muted);margin:0 0 12px}
input[type=email],input[type=password],input[type=text]{width:100%;margin-top:4px;padding:10px 11px;
font:inherit;border:1px solid var(--line);border-radius:2px;background:#fff;color:var(--ink)}
input:focus,button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.hintline{font-size:12px;color:var(--muted);margin:-6px 0 12px}
.show{display:flex;align-items:center;gap:7px;font-size:13px;margin:0 0 14px;cursor:pointer}
.go{width:100%;padding:11px;font:inherit;font-weight:600;color:#fff;background:var(--ink);
border:1px solid var(--ink);border-radius:2px;cursor:pointer}
.go:hover{background:#28394a}.go:disabled{opacity:.55;cursor:wait}
.err{display:none;border-left:3px solid var(--bad);background:var(--badbg);padding:8px 11px;
font-size:13.5px;margin-bottom:14px}
.err.on{display:block}
.note{font-size:12.5px;color:var(--muted);margin-top:14px;line-height:1.5}
.temp{display:none;border-left:3px solid #C07414;background:#FBF3E6;padding:8px 11px;font-size:12.5px;margin-top:14px}
.temp.on{display:block}
.hide{display:none!important}
</style></head><body>
<main class="card">
<div class="brand"><h1>Complaint analyzer</h1>
<p>Sign in to analyze complaints and manage your triage queue.</p></div>
<div class="panel">
<div class="tabs" role="tablist">
<button type="button" role="tab" id="tab_login" aria-selected="true">Sign in</button>
<button type="button" role="tab" id="tab_signup" aria-selected="false">Create account</button></div>
<div class="err" id="err" role="alert"></div>
<form id="f" novalidate>
<label>Email<input id="email" type="email" autocomplete="username" maxlength="254" required autofocus></label>
<label>Password<input id="pw" type="password" autocomplete="current-password" maxlength="128" required></label>
<p class="hintline hide" id="pwhint">At least 8 characters. Avoid common passwords like password123.</p>
<label class="hide" id="confirmrow">Confirm password<input id="pw2" type="password" autocomplete="new-password" maxlength="128"></label>
<label class="show"><input type="checkbox" id="show"> Show password</label>
<button class="go" id="go" type="submit">Sign in</button></form>
<p class="note">Your password is never stored, only a salted hash of it. Customer details you add to the
queue stay in this browser and are not sent to the server.</p>
<div class="temp" id="temp"><b>Demo server.</b> Accounts are kept on temporary storage and are cleared
whenever the app restarts or goes to sleep. If sign-in stops working, create the account again.</div>
</div></main>
<script>
const SIGNUP_ON=__SIGNUP__, TEMP_ACCOUNTS=__TEMP__;
const $=id=>document.getElementById(id);
let mode='login';
const EMAIL=/^[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,190}\.[A-Za-z]{2,24}$/;
function setMode(m){
 if(m==='signup'&&!SIGNUP_ON)return;
 mode=m;const s=m==='signup';
 $('tab_login').setAttribute('aria-selected',String(!s));$('tab_signup').setAttribute('aria-selected',String(s));
 $('confirmrow').classList.toggle('hide',!s);$('pwhint').classList.toggle('hide',!s);
 $('pw').autocomplete=s?'new-password':'current-password';
 $('go').textContent=s?'Create account':'Sign in';
 showErr('');$('pw').value='';$('pw2').value='';
}
function showErr(msg){const e=$('err');e.textContent=msg;e.classList.toggle('on',!!msg);}
$('tab_login').onclick=()=>setMode('login');
$('tab_signup').onclick=()=>setMode('signup');
if(!SIGNUP_ON)$('tab_signup').classList.add('hide');
if(TEMP_ACCOUNTS)$('temp').classList.add('on');
$('show').onchange=()=>{const t=$('show').checked?'text':'password';$('pw').type=t;$('pw2').type=t;};

$('f').addEventListener('submit',async e=>{
 e.preventDefault();showErr('');
 const email=$('email').value.trim().toLowerCase(),pw=$('pw').value;
 if(!EMAIL.test(email)||email.length>254){showErr('Enter a valid email address.');return;}
 if(!pw){showErr('Enter your password.');return;}
 if(mode==='signup'){
  if(pw.length<8){showErr('Use at least 8 characters for the password.');return;}
  if(pw!==$('pw2').value){showErr('The two passwords do not match.');return;}
 }
 $('go').disabled=true;
 try{
  const r=await fetch(mode==='signup'?'/api/signup':'/api/login',{method:'POST',
   headers:{'Content-Type':'application/json'},body:JSON.stringify({email,password:pw})});
  let d={};try{d=await r.json();}catch(_){}
  if(r.ok&&d.ok){location.reload();return;}
  showErr(d.error||'Something went wrong. Please try again.');
 }catch(_){showErr('Could not reach the server. Check your connection and try again.');}
 $('go').disabled=false;
});
</script></body></html>"""
