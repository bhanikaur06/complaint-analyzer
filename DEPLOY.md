# Deploying the app

You'll end up with a public link like `https://complaint-analyzer.onrender.com` that works on any phone or laptop.

Two free accounts needed: **GitHub** (stores the code) and **Render** (runs it). About 15 minutes the first time.

---

## Before you start — one rule

**Don't retrain the model before deploying.**

The model file was saved with scikit-learn 1.8.0, and the server is pinned to exactly that version in `requirements-deploy.txt`. Your laptop has 1.9.0. If you rerun `improve.py` locally, the new model file will be a 1.9.0 file and the server won't load it reliably.

If you do retrain later, open `requirements-deploy.txt` and change `scikit-learn==1.8.0` to whatever `python -c "import sklearn; print(sklearn.__version__)"` prints.

---

## Part A — Put the code on GitHub

1. Go to **github.com** and sign up or log in.
2. Click the **+** at the top right → **New repository**.
3. Name it `complaint-analyzer`. Leave it **Public**. Don't tick any of the "Add a README" boxes. Click **Create repository**.
4. On the next page, click the link **uploading an existing file**.
5. Open File Explorer at `Desktop\project\ml-project`. Select **everything inside** that folder — `data`, `models`, `reports`, `src`, and all the loose files — and drag them onto the GitHub page.

   Drag the *contents*, not the `ml-project` folder itself. GitHub needs `render.yaml` sitting at the top level.

6. Wait for every file to finish uploading. Scroll down, click **Commit changes**.

Check: on your repository's main page you should see `render.yaml`, `src`, `models` and the rest at the top level.

---

## Part B — Run it on Render

1. Go to **render.com** → **Get Started** → sign up **with GitHub**. That links the two accounts.
2. In the Render dashboard click **New +** → **Blueprint**.
3. Pick your `complaint-analyzer` repository. Render finds `render.yaml` and fills in every setting automatically.
4. Render now asks for two values, **DEMO_EMAIL** and **DEMO_PASSWORD**. This is the login you will hand to your teacher.
   - Email: anything that looks like an email, for example `teacher@demo.com`
   - Password: at least 8 characters, not a common one like `password123`. Write it down.

   They are typed into Render, not into the code, so the password never appears on GitHub.
5. Click **Apply** / **Deploy Blueprint**.
6. Watch the log. Installing takes 2–4 minutes. It's finished when you see:

   ```
   Complaint analyzer running at http://localhost:10000
   ```

   (Render shows its own internal port. Ignore it; your public link is what matters.)

7. Your link is at the top of the service page, ending `.onrender.com`. Open it.

---

## Logins and accounts

Anyone who opens your link sees a sign-in page. They can sign in with the demo account, or create their own.

**Accounts do not last on the free plan.** Render's free plan keeps files on temporary storage, and the app goes to sleep after 15 minutes without visitors. When it wakes up, every account anyone created is gone. The demo account is the exception, because the app recreates it from `DEMO_EMAIL` and `DEMO_PASSWORD` on every start.

So for your teacher: give them the demo login, not "create an account". If someone signs up and finds their account missing later, this is why. The sign-in page shows a notice about it on the deployed app.

Making accounts permanent needs a database that survives restarts, which means either a paid plan with a persistent disk or a separate hosted database. That is not built yet.

To close registration so that only the demo login works, add an environment variable `ALLOW_SIGNUP` with the value `0` in Render's dashboard under **Environment**.

What each person's queue holds stays in their own browser, separately for each account. It is not stored on the server, so it does not follow someone to another device.

## Before your demo

**Open the link a minute before you need it.**

Render's free tier puts the app to sleep after 15 minutes with no visitors. The first visit afterwards takes about 50 seconds to wake it up. Every visit after that is instant.

The triage queue is saved in each visitor's own browser, so your teacher won't see your test entries and you won't see theirs.

---

## If it breaks

| What you see | Fix |
|---|---|
| Render can't find a blueprint | `render.yaml` isn't at the top level of the repository. Re-upload the folder's *contents*. |
| Build fails installing scikit-learn | Check `PYTHON_VERSION` in `render.yaml` is still `3.12.3`. |
| "No trained model found" in the log | `models/emotion_final.joblib` didn't upload. Add it on GitHub via **Add file → Upload files** into the `models` folder. |
| Page loads, Analyze does nothing | Wait 60 seconds and refresh — the app was still waking up. |
| Sign-in says the password is wrong right after the app woke up | Normal for accounts people created themselves (see above). The demo account still works. |
| Demo login is refused | `DEMO_PASSWORD` was left blank or is too weak. Check the app's log for "Demo account NOT created". Set it again under **Environment**. |
| "Too many failed attempts" | Five wrong passwords lock that account for 10 minutes. Wait, or restart the app from Render. |
| `InconsistentVersionWarning` in the log | You retrained locally. See the rule at the top. |

---

## Updating it later

Upload changed files on GitHub (**Add file → Upload files**, same names overwrite the old ones). Render notices the change and redeploys by itself within a couple of minutes.
