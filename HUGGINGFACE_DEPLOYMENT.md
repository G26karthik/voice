# 🚀 Deploying to Hugging Face Spaces

## Prerequisites
- Hugging Face account: https://huggingface.co/LuciferMrng
- Git installed on your machine
- Hugging Face CLI (optional but recommended)

---

## Method 1: Deploy via Hugging Face Website (Easiest)

### Step 1: Create a New Space
1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Owner**: LuciferMrng
   - **Space name**: `lecture-voice-to-notes` (or your preferred name)
   - **License**: MIT
   - **Select SDK**: Streamlit
   - **Space hardware**: CPU basic (free) or upgrade to GPU for faster performance
   - **Visibility**: Public

3. Click **"Create Space"**

### Step 2: Upload Files
After creating the space, you'll see an empty repository. Upload these files:

**Required Files:**
```
app.py                  # Main application
requirements.txt        # Dependencies (use requirements_hf.txt content)
README.md              # Documentation (use README_HF.md content)
utils/                 # Folder with all utility modules
  ├── __init__.py
  ├── transcribe.py
  ├── summarize.py
  ├── quiz.py
  ├── exporters.py
  ├── structure.py
  └── models.py
```

**Optional Files:**
```
.streamlit/config.toml  # Streamlit theme configuration
packages.txt           # System packages (for spaCy model)
```

### Step 3: Add Space Metadata
Edit the README.md at the top to include:
```yaml
---
title: Lecture Voice-to-Notes Generator
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
license: mit
---
```

### Step 4: Wait for Build
- Hugging Face will automatically build your space
- Check the "Logs" tab to monitor progress
- First build takes 5-10 minutes (downloading models)
- Once complete, your app will be live!

---

## Method 2: Deploy via Git (Advanced)

### Step 1: Install Hugging Face CLI
```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 2: Create Space via CLI
```bash
huggingface-cli repo create lecture-voice-to-notes --type space --space_sdk streamlit
```

### Step 3: Clone and Push
```bash
# Clone your new space
git clone https://huggingface.co/spaces/LuciferMrng/lecture-voice-to-notes
cd lecture-voice-to-notes

# Copy files from your project
cp -r /path/to/Voice/* .

# Use HF-optimized requirements
cp requirements_hf.txt requirements.txt

# Add metadata to README
cat app_header.yaml README_HF.md > README.md

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

---

## Method 3: Link GitHub Repository (Recommended)

### Step 1: Prepare Repository
Your GitHub repo is already set up at: https://github.com/G26karthik/voice

Add these files to your GitHub repo:

1. **Create a Hugging Face branch:**
```bash
cd c:\Users\saita\OneDrive\Desktop\Projects\Voice

# Copy HF-specific files
cp requirements_hf.txt requirements.txt
cat app_header.yaml README_HF.md > README.md

# Create .streamlit directory if not exists
mkdir -p .streamlit

# Commit HF-specific changes
git checkout -b huggingface
git add .streamlit/config.toml requirements.txt packages.txt
git commit -m "Add Hugging Face Spaces configuration"
git push origin huggingface
```

### Step 2: Link to HF Space
1. Create a new space on Hugging Face
2. Instead of uploading files, use "Import from GitHub"
3. Enter your GitHub repo: `G26karthik/voice`
4. Select branch: `huggingface`
5. Hugging Face will sync automatically

---

## 📋 Required Files Checklist

Create/verify these files before deployment:

- [x] `app.py` - Main Streamlit app
- [x] `requirements.txt` - Python dependencies (use requirements_hf.txt)
- [x] `README.md` - With HF metadata header
- [x] `utils/` - All 6 Python modules
- [ ] `.streamlit/config.toml` - Streamlit configuration (optional)
- [ ] `packages.txt` - System packages (optional, for spaCy)

---

## ⚙️ Configuration Tips

### 1. **Optimize for CPU (Free Tier)**
HF Spaces free tier uses CPU. Your app will work but be slower:
- Transcription: ~2-3x slower than GPU
- Consider using `whisper-tiny` or `whisper-base` for faster performance
- Enable chunking and VAD to reduce processing time

### 2. **Upgrade to GPU (Paid)**
For faster performance:
- Go to Space Settings
- Upgrade hardware to GPU (T4 or A10G)
- Cost: ~$0.60/hour (T4) or ~$3.15/hour (A10G)
- GPU gives 5-10x faster transcription

### 3. **Environment Variables**
Set these in Space Settings if needed:
```
WHISPER_MODEL_SIZE=base        # Use smaller model for CPU
USE_FAST_WHISPER=1             # Enable faster-whisper backend
HF_HOME=/tmp/.cache            # Cache directory
```

### 4. **Memory Management**
Free tier has 16GB RAM. If you encounter memory issues:
- Use smaller Whisper model (`base` or `tiny`)
- Enable chunking for long audio files
- Reduce beam size to 1
- Disable structured notes feature

---

## 🐛 Common Issues & Solutions

### Issue 1: Build Fails - "Out of Memory"
**Solution:** Reduce model sizes in requirements.txt or upgrade to paid tier

### Issue 2: "No module named 'utils'"
**Solution:** Ensure `utils/` folder and `__init__.py` are uploaded

### Issue 3: spaCy model not found
**Solution:** Add this to `packages.txt`:
```
python -m spacy download en_core_web_sm
```

### Issue 4: Slow transcription
**Solution:** 
- Use smaller model: Set `WHISPER_MODEL_SIZE=tiny` in Space settings
- Enable "Fast Mode" in sidebar
- Consider upgrading to GPU hardware

### Issue 5: Audio upload fails
**Solution:** Check file size limits (HF Spaces: 50MB default)

---

## 📊 Expected Performance

### On CPU (Free Tier):
- 5-minute audio: ~3-5 minutes transcription
- 15-minute audio: ~8-12 minutes transcription
- Model: whisper-base or whisper-tiny recommended

### On GPU (T4):
- 5-minute audio: ~30-60 seconds transcription
- 15-minute audio: ~2-3 minutes transcription
- Model: whisper-small or whisper-medium

---

## 🔗 Post-Deployment

After deployment, your space will be available at:
```
https://huggingface.co/spaces/LuciferMrng/lecture-voice-to-notes
```

### Share Your Space:
1. Add description and tags in Space settings
2. Upload a demo video or screenshots
3. Add to your Hugging Face profile
4. Share on social media with #HuggingFace #Streamlit #AI

### Monitor Usage:
- Check "Logs" tab for errors
- Monitor "Analytics" for usage stats
- Update README with demo instructions

---

## 🎯 Quick Start Commands

```bash
# Navigate to project
cd c:\Users\saita\OneDrive\Desktop\Projects\Voice

# Prepare HF deployment files
cp requirements_hf.txt requirements.txt

# Add HF metadata to README (prepend yaml header)
echo "---
title: Lecture Voice-to-Notes Generator
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
license: mit
---
" | cat - README.md > README_temp.md && mv README_temp.md README.md

# Commit changes
git add .
git commit -m "Prepare for Hugging Face Spaces deployment"
git push

# Now upload to HF Spaces via web interface or CLI
```

---

## ✅ Deployment Checklist

Before going live:
- [ ] Test app locally: `streamlit run app.py`
- [ ] Verify all dependencies in requirements.txt
- [ ] Add HF metadata to README.md
- [ ] Upload all files including `utils/` folder
- [ ] Check Space logs for errors
- [ ] Test with sample audio file
- [ ] Add usage instructions to Space README
- [ ] Set appropriate hardware (CPU/GPU)
- [ ] Configure environment variables if needed

---

## 🆘 Need Help?

- HF Spaces Documentation: https://huggingface.co/docs/hub/spaces
- Community Forum: https://discuss.huggingface.co/
- Your GitHub Repo: https://github.com/G26karthik/voice

---

**Your app is ready to deploy! Choose the method that works best for you and follow the steps above.** 🚀
