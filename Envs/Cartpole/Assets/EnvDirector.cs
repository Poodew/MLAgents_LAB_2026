using UnityEngine;
using TMPro;
using System;
using Unity.VisualScripting;

public class EnvDirector : MonoBehaviour
{ 
    [SerializeField] CartController cartController;
    [SerializeField] TMP_Text stateText;

    void Update()
    {
        float cartX = cartController.cartX;
        float cartVel = cartController.cartVel;
        float poleAngle = cartController.poleAngle;
        float poleAngVel = cartController.poleAngVel;

        float currentTime = cartController.currentTime;

        stateText.text =
            $"Cart X: {cartX:F3}\n" +
            $"Cart Vel: {cartVel:F3}\n" +
            $"Pole Angle: {poleAngle:F3}\n" +
            $"Pole AngVel: {poleAngVel:F3}\n\n" +
            $"Time: {currentTime:F3}";
    }
}
