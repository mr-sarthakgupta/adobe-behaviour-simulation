# Adobe's Behaviour Simulation Challenge

### Problem Statement

1. To simulate behaviour (`likes`) from the content of a tweet
2. To simulate content (`tweet text`) from the tweet metadata

### Data Download

[Task 1 Training Data](https://docs.google.com/spreadsheets/d/1JcESl7qCCBvS6xpWMZplhCXunvmkcNU_/edit?usp=drive_link&ouid=101476968084918341858&rtpof=true&sd=true)

[Task 2 Training Data](https://docs.google.com/spreadsheets/d/1oKN_4cMNQHMNrmTSjzKqiJpvDTQA0dAH/edit?usp=drive_link&ouid=101476968084918341858&rtpof=true&sd=true)

**Download and store data in `./data`**

## Task 2

### Training Scripts 

#### 1. Blip inference 
Initial inference of images, videos and gif files for generating captions and prompts to give to LLM based models for finetuning and generating quality captions. 

```
>> python3 ./task2/blip-inference-data.py --data-path=./data/complete_username_data.xlsx --username-data=./data/username_data.csv
```

Note: 
Install these libraries like this to use bits and bytes and other parts used in file 
```
>> pip install -i https://test.pypi.org/simple/ bitsandbytes
>> git clone https://github.com/salesforce/BLIP
```

### References

- [Large Content And Behavior Models To Understand, Simulate, And Optimize Content And Behavior](https://arxiv.org/abs/2309.00359)
