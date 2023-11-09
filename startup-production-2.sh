
echo "-----------Date----------"
date
echo "Configure Nginx"
echo "------------------------------"
sudo cp -r systemd_files/production-2/server/ats-rp-fastapi /etc/nginx/sites-available/ats-rp-fastapi
sudo cp -r systemd_files/production-2/server/ats-rp-flower /etc/nginx/sites-available/ats-rp-flower
sudo systemctl restart nginx
sudo systemctl daemon-reload
sleep 2

echo '[RESTART] ResumeParser API'
sudo cp -r systemd_files/production-2/server/gunicorn.service /etc/systemd/system/gunicorn.service
sudo systemctl restart gunicorn
sleep 2

echo '[RESTART] celery workers'
echo "--------------------------"
sudo cp systemd_files/production-2/celery/celery.conf /etc/supervisor/conf.d/
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart all
sleep 2

echo '[RESTART] Name Service API'
pm2 stop name_service_run
pm2 start systemd_files/production-2/APIs/name_service_run.sh --name name_service_run

echo '[RESTART] NER Service API'
pm2 stop ner_service_run
pm2 start systemd_files/production-2/APIs/ner_service_run.sh --name ner_service_run

echo '[RESTART] Skill Service API'
pm2 stop skill_service_run
pm2 start systemd_files/production-2/APIs/skill_service_run.sh --name skill_service_run

echo '[RESTART] Tika Service API'
pm2 stop tika-server
pm2 start systemd_files/production-2/APIs/tika_service_run.sh --name tika-server

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
