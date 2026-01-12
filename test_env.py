#!/usr/bin/env python3
"""
debug_env.py - Debug environment variable loading
Run this to see what's actually being loaded
"""

import os
from dotenv import load_dotenv

print("=" * 60)
print("ENVIRONMENT VARIABLES DEBUG")
print("=" * 60)
print()

# Check if .env file exists
if os.path.exists('.env'):
    print("✓ .env file found")
    print()
else:
    print("✗ .env file NOT found!")
    print("  Create it by copying .env.example")
    print()
    exit(1)

# Load .env
print("Loading .env file...")
load_dotenv()
print()

# Check critical variables
critical_vars = {
    'AZURE_OPENAI_KEY': 'Azure OpenAI API Key',
    'AZURE_OPENAI_ENDPOINT': 'Azure OpenAI Endpoint',
    'AZURE_OPENAI_DEPLOYMENT': 'Azure OpenAI Deployment',
    'AZURE_SPEECH_KEY': 'Azure Speech Key',
    'AZURE_SPEECH_REGION': 'Azure Speech Region',
    'ARI_PASSWORD': 'ARI Password',
    'SSH_USER': 'SSH Username',
    'SSH_PASSWORD': 'SSH Password',
    'SSH_HOST': 'SSH Host',
}

print("Checking critical variables:")
print("-" * 60)

all_good = True
for var, description in critical_vars.items():
    value = os.getenv(var)
    if value and value.strip():
        # Mask sensitive values
        if 'KEY' in var or 'PASSWORD' in var:
            masked = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '****'
            print(f"✓ {var:<25} = {masked}")
        else:
            print(f"✓ {var:<25} = {value}")
    else:
        print(f"✗ {var:<25} = NOT SET OR EMPTY")
        all_good = False

print()
print("=" * 60)

if all_good:
    print("✓ All critical variables are set!")
else:
    print("✗ Some variables are missing or empty")
    print()
    print("Fix by editing .env file:")
    print("  nano .env")

print()
print("Your .env file contents (without showing secrets):")
print("-" * 60)

with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            if '=' in line:
                key, value = line.split('=', 1)
                if 'KEY' in key or 'PASSWORD' in key:
                    if value:
                        print(f"{key}=****[SET]")
                    else:
                        print(f"{key}=[EMPTY]")
                else:
                    print(line)

print("=" * 60)