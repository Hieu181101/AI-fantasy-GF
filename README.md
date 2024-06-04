# AI-Fantasy-GF
So yah being a CS student meaning that I do not have the time to touch grass and after watching the move "Her" I really want an AI GirlFriend. So I decide why not make one ? Who needs a real Girlfriend anyway

## The Goals:
- She should be able to run with a decent GPU
- She can understand me well enough where if I ask about what is the distance to the moon she would not talk about sushi
- I can text to her and speak to her
- She can text back and speak to me
- I can see my GF
- I can change her apperance

## Communication is KEY (in any relationship) 
First she kinda needs a brain where she can talk to me. This step is quite easy with all the AI model like ChatGPT or Gemini; we can just use those for our GF brain. I will use the model 3 turbo from Open AI since I want my GF to be super smart but feel free to change. 
And adding my perfect imagination of what my GF would behave to the Prompt and Walla she can now speak to me. (just through text only first)
One more thing: She kinda needs to remeber what we are talking about so I implement a History holder that would retain the conversation we have 

Let's move on to how she can hear me and be able to speak back: This part is easy enough; we can use TTS and STT for our communication. 
I mess around with two model: 1. tacotron2 and 2. XTTS-v2. From what I can see they do their job nicely. tacotron does not need a strong GPU to use and give a quite decent results. However, if you want your GF to mimic a voices you like XTTS-v2 is a great option. 
Perfect now we can hear and talk to each others. 

## See my Gf and change her appearance 

- **Seeing my GF**: Ok having someone to talks to is quite fun but it do not want people to judge me speaking to my computer so having an option to create an image for my gf would be nice. What comme to mind would be somesort of diffusion models.
Luckily Hugging Face got us and I am able to find "waifu-diffusion v1.4 - Diffusion for Weebs" yah ignore that name. Perfect now I can actually see my GF

- **Change her Appearance**: I love Dnd and and I always ask myself what it would be like to be able to date other racecs from Dnd. And yah I did just that. I can now change my GF apperance to Elf, Cat Girl or Dwaft 



## Look Through
![Screenshot (3)](https://github.com/Hieu181101/MAZA-visulization/assets/135567323/bcac7ded-c160-42b3-bc8e-e3994f4de9d5)

![Screenshot (10)](https://github.com/Hieu181101/MAZA-visulization/assets/135567323/af1d346e-8d36-42ba-9e7f-d909bb1bc936)

![Screenshot (6)](https://github.com/Hieu181101/MAZA-visulization/assets/135567323/1c7e18b0-43a4-43f0-b3fb-2a0ce732a1aa)



