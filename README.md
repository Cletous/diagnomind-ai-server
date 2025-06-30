# DiagnoMind – AI-Powered Medical Diagnosis System

[![Laravel](https://img.shields.io/badge/Laravel-12.x-red?logo=laravel)](https://laravel.com/)
[![Python](https://img.shields.io/badge/Flask-Python-blue?logo=python)](https://flask.palletsprojects.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-lightgrey?logo=mysql)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DiagnoMind is a full-stack medical diagnosis system built with **Laravel + Livewire** and integrated with an **AI diagnosis server using Flask (Python)**. Doctors can submit symptoms, receive AI-generated suggestions, manage hospitals, and track patient feedback. Patients can access diagnosis history and provide ratings or comments.

---

## Getting Started (Development Setup)

### 1. Create a folder diagnomind-projects

You are going to need to Extract or clone the project folders into a file structure as follows:

```

diagnomind-projects/
├── diagnomind-ai-server/ # Python Flask AI engine
└── diagnomind.localhost/ # Laravel web frontend

```

- To Clone Python Flask AI engine use: `git clone https://github.com/Cletous/diagnomind-ai-server`
- To Clone Laravel web frontend use: `git clone https://github.com/Cletous/diagnomind.localhost`

---

### 2. Start the AI Diagnosis Server

```bash
cd diagnomind-projects/diagnomind-ai-server
python app.py
```

This runs the Flask server at: `http://127.0.0.1:2500/predict`

---

### 3. Configure Laravel Environment

```bash
cd ../diagnomind.localhost
cp .env.example .env
```

Edit `.env` and configure your database for example:

```dotenv
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=diagnomind
DB_USERNAME=root
DB_PASSWORD=
```

---

### 4. Install PHP Dependencies

```bash
composer install
```

---

### 5. Run Migrations and Seeders

```bash
php artisan migrate:fresh --seed
```

---

### 6. Default User Accounts

| Role         | Email                                | Password  |
| ------------ | ------------------------------------ | --------- |
| **Admin**    | `admin@test.com`                     | `admin`   |
| **Doctors**  | `doctor1@test.com`                   | `doctor`  |
|              | `doctor2@gmail.com`                  | `doctor`  |
| **Patients** | `user1@test.com` → `user20@test.com` | `patient` |

---

### 7. Link and configure your mail provider:

```dotenv
MAIL_MAILER=smtp
MAIL_HOST=server1.makuruwan.com
MAIL_PORT=465
MAIL_USERNAME=no-reply@yourdomain.com
MAIL_PASSWORD=EmailPassword
MAIL_ENCRYPTION=ssl
MAIL_FROM_ADDRESS="no-reply@yourdomain.com"
MAIL_FROM_NAME="${APP_NAME}"
```

Replace `MAIL_HOST, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD, MAIL_ENCRYPTION` and `MAIL_FROM_ADDRESS` with your actual mail provider configuration settings and credentials.

---

### 8. Start Laravel Development Server

```bash
php artisan serve
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Key Features

- Role-based Access (Admin, Doctor, Patient)
- AI Symptom-to-Diagnosis engine (via Flask)
- Hospital Management & Invitations
- Diagnosis History & Filtering
- Feedback & Ratings for Diagnoses

---

## Tech Stack

- **Backend:** Laravel 12, Livewire, MySQL
- **AI Engine:** Flask + Python
- **UI:** Bootstrap 5.3
- **Auth:** Email Verification, Role Management

---

## Contributing

Contributions are welcome! Fork the repo and submit a PR.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Maintained by **UZ Computer Engineering Students** — Part 2 Group Project
**Group Members:**

- Cletous Ngoma (R196481X) – [ngomacletousjnr@gmail.com](mailto:ngomacletousjnr@gmail.com)
- Takudzwanashe H. Nhaiso (R231702X)
- Talent Nechitukire (R1810926)
- Shelton Mutambirwa (R231684F)
- Patricia Mukunza (R231733N)
- Tadiwa Ncube (R231692S)

---
