# device-monitor-prototype



### usage


```shell
python3 -m venv .
source ./bin/activate 
python3 -m pip install -r requirements.txt

export SAUCE_ACCESS_KEY=""
export SAUCE_USERNAME=""
export SAUCE_REGION=eu-central-1

mkdir -p data
python device_monitor.py --debug \
    --region ${SAUCE_REGION} \
    --data-file ./data/device_monitoring-${SAUCE_REGION}-${SAUCE_USERNAME}.json

# menu navigation
# q to quit
#
# views
# 1 for status
# 2 for state
# 3 for stats
# 
# scope
# a for all
# h for hour
# w for week


deactivate
```


---
![Alt Text](./device-monitor.gif)

