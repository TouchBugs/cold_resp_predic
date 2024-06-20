import subprocess
import yagmail
import time
import psutil

def send_email(subject, body):
    # 1439389719
    qq = 2196692208
    if len(str(qq))!=10:
        raise ValueError('qq号码长度不对')
    receiver = str(qq) +'@qq.com'  # 接收方邮箱
    yag = yagmail.SMTP(user='2196692208@qq.com', host='smtp.qq.com', port=465, smtp_ssl=True) 
    yag.send(to=receiver, subject=subject, contents=body)
    print('send email successfully')

def is_process_running(process_name):
    """检查是否有指定名称的进程在运行"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if process_name in proc.info['name']:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def monitor_script(script_path, process_name):
    try:
        print(f"Starting the script: {script_path}")
        process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        while True:
            # 检查主程序进程是否仍在运行
            if not is_process_running(process_name):
                print(f"Main script {process_name} is not running. Stopping monitor.")
                # send_email('Task Terminated', f'The monitored script {script_path} has stopped or does not exist.')
                break

            retcode = process.poll()
            if retcode is not None:
                # 主程序已完成或出现异常
                stdout, stderr = process.communicate()
                if retcode == 0:
                    print("Script completed successfully.")
                    send_email('Task Completed', 'Your task has completed successfully.\n\nOutput:\n' + stdout)
                else:
                    print(f"Script failed with return code {retcode}")
                    send_email('Task Failed', f'Your task failed with return code {retcode}.\n\nError output:\n' + stderr)
                break

            time.sleep(10)  # 每10秒检查一次

    except Exception as e:
        print(f"Failed to start the script: {e}")
        # send_email('Monitoring Failed', f'Failed to monitor the script with error: {e}')

# 调用监控函数，传入要监控的脚本路径和主程序名称
monitor_script('/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/test.py', 'test')