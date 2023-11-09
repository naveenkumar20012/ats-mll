
echo "-----------Date----------"
date
echo '[RESTART] ResumeParser API'
sudo systemctl restart gunicorn
sleep 2

echo '[RESTART] Nginx'
sudo systemctl restart nginx
sleep 2

echo '[RESTART] celery workers'
sudo supervisorctl restart all
sleep 2

echo '[RESTART] Name Service API'
pm2 stop name_service_run
pm2 start name_service_run.sh --name name_service_run

echo '[RESTART] NER Service API'
pm2 stop ner_service_run
pm2 start ner_service_run.sh --name ner_service_run

echo '[RESTART] JD Service API'
pm2 stop jd_service_run
pm2 start jd_service_run.sh --name jd_service_run

echo '[RESTART] Score Service API'
pm2 stop score_service_run
pm2 start 'systemd_files/staging/APIs/score_service_run.sh' --name score_service_run

echo "-----------Free memory----------"
free -mh
echo "-----------Free memory----------"

sleep 2

echo "-----------Disk Usage----------"
df -i
echo "-----------Disk Usage----------"

echo
echo -n "  System information as of "
/bin/date
echo
/usr/bin/landscape-sysinfo
