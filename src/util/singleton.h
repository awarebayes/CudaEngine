//
// Created by dev on 7/11/22.
//

#ifndef COURSE_RENDERER_SINGLETON_H
#define COURSE_RENDERER_SINGLETON_H

template <typename T>
class SingletonCreator
{
private:
public:
	std::shared_ptr<T> get()
	{
		static std::shared_ptr<T> singleton = nullptr;
		if (!singleton)
			singleton = std::make_shared<T>();
		return singleton;
	}
};

#endif//COURSE_RENDERER_SINGLETON_H
