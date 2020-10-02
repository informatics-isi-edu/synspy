import os
import subprocess
from deriva.core import format_exception
from launcher.impl import LauncherTask, Task


class SubprocessTask(LauncherTask):
    def __init__(self, parent=None):
        super(SubprocessTask, self).__init__(parent)


class ViewerTask(SubprocessTask):
    def __init__(self, executable, is_owner, proc_output_path=None, parent=None):
        super(SubprocessTask, self).__init__(parent)
        self.executable = executable
        self.is_owner = is_owner
        self.proc_output_path = proc_output_path

    def result_callback(self, success, result):
        self.set_status(success,
                        "Viewer subprocess execution success" if success else "Viewer subprocess execution failed",
                        "" if success else format_exception(result),
                        self.is_owner)

    def run(self, file_path, working_dir=os.getcwd(), env=None):
        self.task = Task(self._execute,
                         [self.executable, file_path, working_dir, self.proc_output_path, env],
                         self.result_callback)
        self.start()

    @staticmethod
    def _execute(executable, file_path, working_dir, proc_output_path=None, env=None):
        out = subprocess.PIPE
        if proc_output_path:
            try:
                out = open(proc_output_path, "wb")
            except OSError:
                pass
        command = [executable, file_path]
        process = subprocess.Popen(command,
                                   cwd=working_dir,
                                   env=env,
                                   stdin=subprocess.PIPE,
                                   stdout=out,
                                   stderr=subprocess.STDOUT)
        ret = process.wait()
        try:
            out.flush()
            out.close()
        except:
            pass
        del process
        if ret != 0:
            raise RuntimeError('Non-zero viewer exit status %s!' % ret)

