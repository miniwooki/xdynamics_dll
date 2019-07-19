#ifndef XVANIMATIONCONTROLLER_H
#define XVANIMATIONCONTROLLER_H

#define MAX_FRAME 1000

class xvAnimationController
{
public:
	xvAnimationController();
	~xvAnimationController();

	static bool is_end_frame();
	static void move2previous2x();
	static void move2previous1x();
	static void on_play();
	static bool Play();
	static void off_play();
	static void move2forward1x();
	static void move2forward2x();
	static void update_frame();
	static void moveStart() { current_frame = 0; }
	static void moveEnd() { current_frame = buffer_count; }
	static void initFrame() { current_frame = 0; }
	static unsigned int getTotalBuffers() { return buffer_count; }
	static void setFrame(unsigned int f) { current_frame = f; }
	static void setPlayMode(int m) { play_mode = m; }
	static int getFrame() { return (int)current_frame; }
	static void upBufferCount() { buffer_count++; }
	static void allocTimeMemory(unsigned int sz) { times = new float[sz]; }
	static void addTime(unsigned int i, float time) { times[i] = time; }
	static float getTime() { return times[current_frame]; }
	static float& getTimes(unsigned int i) { return times[i]; }
	static void setTotalFrame(unsigned int tf) { buffer_count = tf; }
	static void setRealTimeParameter(bool rt) { real_time = rt; }
	static bool getRealTimeParameter() { return real_time; }
	static int currentPlayMode() { return play_mode; }
	static void releaseTimeMemory();

private:
	static int play_mode;
	static unsigned int current_frame;
	static unsigned int buffer_count;
	static bool is_play;
	static bool real_time;
	static float *times;
};


#endif