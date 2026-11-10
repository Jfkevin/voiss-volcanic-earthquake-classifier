# Foto profil privat untuk halaman About VECAS

Foto profil tidak perlu dimasukkan ke repository GitHub.

1. Siapkan foto JPG yang sudah dipotong dan dikompres (disarankan lebar 600 px, ukuran di bawah 150 KB).
2. Jalankan PowerShell:
   ```powershell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\foto-profil.jpg")) | Set-Clipboard
   ```
3. Buka Streamlit Community Cloud → aplikasi VECAS → Manage app → Settings → Secrets.
4. Tambahkan:
   ```toml
   profile_photo_base64 = "PASTE_BASE64_DI_SINI"
   ```
5. Simpan secrets dan reboot aplikasi.

Foto tidak disimpan pada GitHub. Namun, karena foto ditampilkan di halaman About publik, pengunjung website tetap dapat melihat dan menyimpannya.
